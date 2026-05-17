"""
eSEN pretraining from random initialization on our ASE DB datasets.

Architecture: eSCNMDBackbone + MLP_EFS_Head, built directly without loading
any pretrained checkpoint. Conservative forces via autograd (direct_forces=False,
regress_forces=True). EFPNorm replaces EquivariantRMSNormArraySphericalHarmonicsV2
in all 9 norm positions after backbone construction.

Our ASE DBs store atomization energies (DFT total energy minus per-element
atomic energies), so no element-reference conversion is needed here —
the targets are already in a normalized chemical space.

Usage:
    python train/pretrain.py --dataset aimnet2 --epochs 30 --lr 4e-4
    python train/pretrain.py --dataset aimnet2 --max_train_frames 50000 --max_val_frames 5000
    python train/pretrain.py --dataset all --epochs 10 --lr 2e-4 --no_efp_norm
"""

import argparse
import time
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from ase.db import connect
from torch.utils.data import Dataset, DataLoader
from torch_cluster import radius_graph

from fairchem.core.models.uma.escn_md import eSCNMDBackbone, MLP_EFS_Head
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.models.uma.nn.layer_norm import EquivariantRMSNormArraySphericalHarmonicsV2

import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent.parent))
from model.efpnorm import EquivariantEFPNorm

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────

_ROOT   = Path(__file__).parent.parent
DATA_DIR = _ROOT / "data" / "asedb"
DATASETS = ["aimnet2", "spice2", "qdpi", "ani2x", "spf"]

# ── Config ────────────────────────────────────────────────────────────────────

# Verified against the eSEN-SM conserving checkpoint's model_config.
# All architectural parameters match so scratch-trained models are directly
# comparable to the pretrained baseline.
#
# Architectural (change = different weight shapes, checkpoints incompatible):
#   num_layers, lmax/mmax, sphere_channels, hidden_channels, edge_channels,
#   num_distance_basis, ff_type, norm_type, act_type
#
# Dataset/deployment context (must match at inference, but don't change weight shapes):
#   cutoff, max_num_elements, max_neighbors, use_dataset_embedding
#
# Optimization-only (free to tune, see bottom of file):
#   lr, batch_size, epochs, clip_grad_norm, FORCE_COEF, ENERGY_COEF
BACKBONE_CFG = dict(
    # ── architecture ──────────────────────────────────────────────────────────
    num_layers          = 4,
    lmax                = 2,
    mmax                = 2,
    sphere_channels     = 128,
    hidden_channels     = 128,
    edge_channels       = 128,
    num_distance_basis  = 64,         # official eSEN-SM value (NOT 512)
    distance_function   = "gaussian",
    ff_type             = "spectral",  # official eSEN-SM value (NOT "grid")
    norm_type           = "rms_norm_sh",
    act_type            = "gate",
    chg_spin_emb_type   = "rand_emb",  # official eSEN-SM value (NOT "pos_emb")
    # ── conservative force settings ───────────────────────────────────────────
    direct_forces       = False,       # F = -∇E via autograd
    regress_forces      = True,
    regress_stress      = False,
    # ── dataset context ───────────────────────────────────────────────────────
    cutoff              = 6.0,         # Å — must match graph construction below
    max_num_elements    = 100,
    use_dataset_embedding = False,     # single-task training, no dataset token
    otf_graph           = False,       # we supply edge_index in collate_fn
)

CUTOFF     = BACKBONE_CFG["cutoff"]
MAX_ATOMS  = 50
BATCH_SIZE = 8
LR         = 4e-4      # standard from-scratch rate (fine-tuning uses 2e-5)
EPOCHS     = 30
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SEED       = 42

FORCE_COEF  = 1.0
ENERGY_COEF = 0.01    # small weight: atomization energies need no ref conversion

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


# ── Model construction ────────────────────────────────────────────────────────

class eSEN(nn.Module):
    """eSCNMDBackbone + MLP_EFS_Head wired for conservative force prediction.

    Accepts a cfg dict so the caller controls architecture without relying on
    module-level globals. MLP_EFS_Head reads sphere_channels, hidden_channels,
    and regress_config directly from the backbone it receives, so it doesn't
    need a separate config.

    Conservative force path:
      backbone marks pos.requires_grad_(True) → MLP_EFS_Head accumulates
      per-node energies in float64 → forces = -∇_pos(energy) via autograd,
      with create_graph=training so force-loss gradients flow back through
      the differentiation step into the backbone weights.
    """
    def __init__(self, cfg: dict):
        super().__init__()
        self.backbone = eSCNMDBackbone(**cfg)
        self.head = MLP_EFS_Head(self.backbone, wrap_property=False)

    def forward(self, data: AtomicData) -> dict:
        return self.head(data, self.backbone(data))


def replace_norms_with_efp(model: nn.Module) -> int:
    """Swap all EquivariantRMSNorm layers for EquivariantEFPNorm in-place.

    For scratch training, the backbone's norm layers are freshly initialized
    (affine_weight=ones, affine_bias=zeros), so the copy step preserves those
    defaults. The only difference at init is the scale formula:
      standard: 1 / sqrt(rms^2 + eps)  with eps=1e-5 (near-zero → large scale)
      EFP:      1 / sqrt(rms^2 + c^2)  with c≈1.0  (never larger than 1/c)
    EFP is actually safer at random init where features are small.
    """
    replacements = [
        name for name, mod in model.named_modules()
        if isinstance(mod, EquivariantRMSNormArraySphericalHarmonicsV2)
    ]
    for name in replacements:
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        old_mod = getattr(parent, parts[-1])
        new_mod = EquivariantEFPNorm(
            lmax=old_mod.lmax,
            num_channels=old_mod.num_channels,
            affine=old_mod.affine,
            normalization=old_mod.normalization,
            centering=old_mod.centering,
            std_balance_degrees=old_mod.std_balance_degrees,
        )
        if old_mod.affine_weight is not None:
            new_mod.affine_weight.data.copy_(old_mod.affine_weight.data)
        if old_mod.affine_bias is not None:
            new_mod.affine_bias.data.copy_(old_mod.affine_bias.data)
        setattr(parent, parts[-1], new_mod)
    return len(replacements)


# ── Dataset ───────────────────────────────────────────────────────────────────

class AseDbDataset(Dataset):
    """Loads an ASE SQLite DB.

    max_frames caps rows after the max_atoms filter — the loop breaks early
    so large DBs never need to be fully scanned.
    """
    def __init__(self, db_path: Path, cutoff: float, max_atoms: int = MAX_ATOMS,
                 max_frames: int | None = None):
        self.cutoff    = cutoff
        self.max_atoms = max_atoms
        rows = []
        with connect(str(db_path)) as db:
            for row in db.select():
                atoms = row.toatoms()
                if len(atoms) <= max_atoms:
                    rows.append((
                        atoms.numbers.copy(),
                        atoms.positions.copy().astype(np.float32),
                        float(atoms.get_potential_energy()),
                        atoms.get_forces().copy().astype(np.float32),
                    ))
                    if max_frames is not None and len(rows) >= max_frames:
                        break
        self.rows = rows
        cap = f"/{max_frames}" if max_frames is not None else ""
        print(f"[Dataset] {db_path.name}: {len(self.rows)}{cap} frames")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        Z, pos, E, F = self.rows[idx]
        return (
            torch.tensor(Z,   dtype=torch.long),
            torch.tensor(pos, dtype=torch.float32),
            torch.tensor(E,   dtype=torch.float32),
            torch.tensor(F,   dtype=torch.float32),
        )


def collate_fn(batch):
    """Pack variable-size molecules into a single batched AtomicData.

    The backbone requires:
      - pos, atomic_numbers, batch    — core graph data
      - cell, pbc, cell_offsets       — set to zero/False for non-periodic molecules
      - natoms, nedges                — used for energy/stress reduction
      - charge, spin                  — charge=0 (neutral), spin=1 (singlet)
      - fixed, tags                   — zeros (not used in our datasets)
      - edge_index                    — pre-built radius graph; otf_graph=False means
                                        backbone expects this, not the PBC graph builder
    """
    Zs, poss, Es, Fs = zip(*batch)
    n_graphs  = len(batch)
    batch_idx = torch.cat([torch.full((len(z),), i, dtype=torch.long) for i, z in enumerate(Zs)])
    Z_cat     = torch.cat(Zs)
    pos_cat   = torch.cat(poss)
    E_cat     = torch.stack(Es)
    F_cat     = torch.cat(Fs)

    edge_index       = radius_graph(pos_cat, r=CUTOFF, batch=batch_idx, loop=False)
    n_edges          = edge_index.shape[1]
    nedges_per_graph = torch.bincount(batch_idx[edge_index[0]], minlength=n_graphs)

    return AtomicData(
        pos            = pos_cat,
        atomic_numbers = Z_cat,
        cell           = torch.zeros(n_graphs, 3, 3),
        pbc            = torch.zeros(n_graphs, 3, dtype=torch.bool),
        natoms         = torch.tensor([len(z) for z in Zs]),
        edge_index     = edge_index,
        cell_offsets   = torch.zeros(n_edges, 3),
        nedges         = nedges_per_graph,
        charge         = torch.zeros(n_graphs, dtype=torch.long),
        spin           = torch.ones(n_graphs, dtype=torch.long),
        fixed          = torch.zeros(len(Z_cat), dtype=torch.long),
        tags           = torch.zeros(len(Z_cat), dtype=torch.long),
        energy         = E_cat,
        forces         = F_cat,
        batch          = batch_idx,
        sid            = [str(i) for i in range(n_graphs)],
        dataset        = None,
    )


def move_to(data: AtomicData, device):
    tensor_keys = [
        "pos", "atomic_numbers", "cell", "pbc", "natoms",
        "edge_index", "cell_offsets", "nedges",
        "charge", "spin", "fixed", "tags",
        "energy", "forces", "batch",
    ]
    for k in tensor_keys:
        v = getattr(data, k, None)
        if v is not None:
            setattr(data, k, v.to(device))
    return data


# ── Loss ──────────────────────────────────────────────────────────────────────

def loss_fn(pred: dict, data: AtomicData, device: str) -> torch.Tensor:
    """L1 force loss + L1 energy-per-atom loss.

    Energy reference note: our DBs store atomization energies. The model
    predicts a sum of per-atom scalars from random init, which starts near
    zero. The target is already reference-subtracted so no element_references
    conversion is needed (unlike the fine-tuning script where we had to
    convert from the checkpoint's reference frame).

    force loss: L1 over all atom-component pairs — robust to outliers
    energy loss: L1 per-atom (divided by natoms) — scale-invariant across
                 molecules of different sizes
    """
    pred_F = pred["forces"].float() if torch.is_tensor(pred["forces"]) \
        else pred["forces"]["forces"].float()
    force_loss = nn.functional.l1_loss(pred_F, data.forces.to(device))
    total = FORCE_COEF * force_loss

    if ENERGY_COEF > 0.0:
        pred_E = pred["energy"].float() if torch.is_tensor(pred["energy"]) \
            else pred["energy"]["energy"].squeeze(-1).float()
        natoms = data.natoms.float().to(device)
        energy_loss = nn.functional.l1_loss(
            pred_E / natoms, data.energy.to(device) / natoms
        )
        total = total + ENERGY_COEF * energy_loss

    return total


# ── Train / Eval ──────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, device, train: bool):
    model.train(train)
    total_loss = total_fmae = total_steps = 0

    # torch.enable_grad() is required even during validation because the
    # conservative head always differentiates energy w.r.t. positions.
    # Using torch.no_grad() here would break force computation entirely.
    with torch.enable_grad():
        for data in loader:
            data = move_to(data, device)

            # The backbone internally calls data["pos"].requires_grad_(True)
            # when direct_forces=False and regress_forces=True, but we set it
            # here too for safety (e.g. if the data object was cloned somewhere).
            data.pos.requires_grad_(True)

            if train:
                optimizer.zero_grad()

            out = model(data)
            loss = loss_fn(out, data, device)

            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 100.0)
                optimizer.step()

            raw_F = out["forces"] if torch.is_tensor(out["forces"]) \
                else out["forces"]["forces"]
            f_mae = (raw_F.detach().float() - data.forces).abs().mean().item()
            total_loss  += loss.item()
            total_fmae  += f_mae
            total_steps += 1

    return total_loss / total_steps, total_fmae / total_steps


# ── Main ──────────────────────────────────────────────────────────────────────

def build_datasets(args):
    names = DATASETS if args.dataset == "all" else [args.dataset]
    train_list, val_list = [], []
    for name in names:
        tr_db = DATA_DIR / f"{name}_train.db"
        va_db = DATA_DIR / f"{name}_val.db"
        if not tr_db.exists():
            print(f"[{name}] SKIP — {tr_db} not found")
            continue
        train_list.append(AseDbDataset(tr_db, CUTOFF, max_atoms=args.max_atoms,
                                       max_frames=args.max_train_frames))
        val_list.append(AseDbDataset(va_db, CUTOFF, max_atoms=args.max_atoms,
                                     max_frames=args.max_val_frames))
    return train_list, val_list


class ConcatDataset(torch.utils.data.ConcatDataset):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS + ["all"], default="aimnet2")
    parser.add_argument("--epochs",       type=int,   default=EPOCHS)
    parser.add_argument("--batch_size",   type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",           type=float, default=LR)
    parser.add_argument("--max_atoms",    type=int,   default=MAX_ATOMS)
    parser.add_argument("--resume",       type=str,   default=None,
                        help="Resume from a previous pretrain checkpoint")
    parser.add_argument("--out",          type=str,   default=None,
                        help="Output checkpoint path (default: scratch/pretrain_<dataset>.pt)")
    parser.add_argument("--no_efp_norm",  action="store_true",
                        help="Disable EFPNorm (use original EquivariantRMSNorm)")
    parser.add_argument("--max_train_frames", type=int, default=None)
    parser.add_argument("--max_val_frames",   type=int, default=None)
    # Architectural overrides — changing these produces a different model
    # that cannot load checkpoints from the default configuration.
    # hidden_channels and edge_channels default to None, which auto-couples
    # them to sphere_channels (matching every official eSEN-SM/MD/LG config).
    # Pass explicit values only to decouple widths intentionally.
    parser.add_argument("--num_layers",      type=int,          default=BACKBONE_CFG["num_layers"])
    parser.add_argument("--sphere_channels", type=int,          default=BACKBONE_CFG["sphere_channels"])
    parser.add_argument("--hidden_channels", type=int,          default=None,
                        help="Defaults to sphere_channels if not set (eSEN design)")
    parser.add_argument("--edge_channels",   type=int,          default=None,
                        help="Defaults to sphere_channels if not set (eSEN design)")
    args = parser.parse_args()

    # Resolve coupled defaults: all three widths track sphere_channels unless
    # explicitly decoupled. This matches every official eSEN variant.
    hidden_channels = args.hidden_channels if args.hidden_channels is not None else args.sphere_channels
    edge_channels   = args.edge_channels   if args.edge_channels   is not None else args.sphere_channels

    out_ckpt = Path(args.out) if args.out else _ROOT / "scratch" / f"pretrain_{args.dataset}.pt"

    print(f"Device: {DEVICE}")
    print(f"\nLoading datasets ...")
    train_list, val_list = build_datasets(args)
    if not train_list:
        raise RuntimeError("No datasets found")

    train_ds = ConcatDataset(train_list) if len(train_list) > 1 else train_list[0]
    val_ds   = ConcatDataset(val_list)   if len(val_list)   > 1 else val_list[0]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    print(f"\nBuilding eSEN from scratch ...")
    cfg = {
        **BACKBONE_CFG,
        "num_layers":      args.num_layers,
        "sphere_channels": args.sphere_channels,
        "hidden_channels": hidden_channels,
        "edge_channels":   edge_channels,
    }
    print(f"  num_layers={cfg['num_layers']}  "
          f"sphere/hidden/edge={cfg['sphere_channels']}/{cfg['hidden_channels']}/{cfg['edge_channels']}")
    model = eSEN(cfg)

    if not args.no_efp_norm:
        n = replace_norms_with_efp(model)
        print(f"  EFPNorm: replaced {n} EquivariantRMSNorm layers")

    model = model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params/1e6:.1f}M parameters")

    start_epoch = 1
    if args.resume:
        state = torch.load(args.resume, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state["model"])
        start_epoch = state["epoch"] + 1
        print(f"  Resumed from epoch {state['epoch']}: {args.resume}")

    # no_weight_decay() returns a set of parameter names whose norms/biases
    # should not be penalized. eSCNMDBackbone implements this.
    no_wd = set(model.backbone.no_weight_decay()) if hasattr(model.backbone, "no_weight_decay") else set()
    params_no_wd, params_wd = [], []
    for name, param in model.named_parameters():
        if any(name.endswith(k) for k in no_wd):
            params_no_wd.append(param)
        else:
            params_wd.append(param)
    optimizer = torch.optim.AdamW(
        [{"params": params_wd, "weight_decay": 1e-3},
         {"params": params_no_wd, "weight_decay": 0.0}],
        lr=args.lr,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )
    for _ in range(start_epoch - 1):
        scheduler.step()

    print(f"\nPretraining for {args.epochs} epochs (LR={args.lr:.2e}) ...")
    best_val_fmae = float("inf")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_fmae = run_epoch(model, train_loader, optimizer, DEVICE, train=True)
        va_loss, va_fmae = run_epoch(model, val_loader,   optimizer, DEVICE, train=False)
        scheduler.step()
        dt = time.time() - t0

        star = " *" if va_fmae < best_val_fmae else ""
        if va_fmae < best_val_fmae:
            best_val_fmae = va_fmae
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "backbone_cfg": cfg}, out_ckpt)

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train loss={tr_loss:.4f}  F-MAE={tr_fmae*1000:.1f}meV/Å  |  "
            f"val loss={va_loss:.4f}  F-MAE={va_fmae*1000:.1f}meV/Å  "
            f"{dt:.1f}s{star}"
        )

    print(f"\nBest val F-MAE: {best_val_fmae*1000:.1f} meV/Å")
    print(f"Checkpoint saved → {out_ckpt}")


if __name__ == "__main__":
    main()
