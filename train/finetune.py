"""
Fine-tune a pretrained eSEN checkpoint on our molecular h5 datasets.

Uses initialize_finetuning_model to load the eSEN-SM conserving checkpoint
(6.3M params, 4 layers, conservative forces) and fine-tunes on our ASE DBs.

The pretrained model uses raw total DFT energies with per-element linear
references. Our data stores atomization energies (E_dft - sum_Z ref_Z).
The checkpoint's element references are extracted and applied to our targets
so both sides are in the same convention: model output ≈ E_dft - checkpoint_refs.

Usage:
    python train/finetune.py --dataset ani2x --epochs 10 --lr 2e-5
    python train/finetune.py --dataset all   --epochs 5  --lr 1e-5 --freeze_layers 2
    python train/finetune.py --dataset aimnet2 --max_train_frames 20000 --max_val_frames 2000
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

from fairchem.core.units.mlip_unit.mlip_unit import initialize_finetuning_model
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.models.uma.nn.layer_norm import EquivariantRMSNormArraySphericalHarmonicsV2

import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent.parent))
from model.efpnorm import EquivariantEFPNorm

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).parent.parent

CKPT_PATH = _ROOT / (
    "scratch/fairchem_cache/models--facebook--OMol25/"
    "snapshots/039b7070e59d1537e56c93a3a455263d062ed9c8/"
    "checkpoints/esen_sm_conserving_all.pt"
)

DATA_DIR = Path(__file__).parent.parent / "data" / "asedb"

DATASETS = ["aimnet2", "spice2", "qdpi", "ani2x", "spf"]

# ── Config ────────────────────────────────────────────────────────────────────

CUTOFF     = 6.0      # Å — must match the pretrained model
MAX_ATOMS  = 50       # skip large molecules to keep batches tractable
BATCH_SIZE = 8
LR         = 2e-5     # much lower than training from scratch (4e-4)
EPOCHS     = 10
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SEED       = 42

FORCE_COEF  = 1.0
ENERGY_COEF = 0.0     # disabled until ref-frame correction is verified per dataset

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


# ── Element references (from pretrained checkpoint) ───────────────────────────

def load_element_refs(ckpt_path: Path) -> torch.Tensor:
    """
    Return a float64 tensor of shape [max_Z+1] with linear reference energies
    (eV) from the pretrained checkpoint's tasks_config.

    Our atomization energies = E_dft - sum_Z our_refs[Z].
    Checkpoint model output  ≈ E_dft - sum_Z ckpt_refs[Z].
    So target for fine-tuning = E_atomization + sum_Z our_refs[Z] - sum_Z ckpt_refs[Z]
                               = E_dft        - sum_Z ckpt_refs[Z]

    Reference energies can be very large (e.g. C ≈ −1037 eV), so the correction
    is NOT small — it must be applied accurately before the energy loss is used.
    (Energy loss is currently disabled via ENERGY_COEF=0.0 until ref frames are
    verified per dataset.)
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    tasks_config = ckpt.tasks_config if hasattr(ckpt, "tasks_config") else ckpt["tasks_config"]
    print(f"  checkpoint type : {type(ckpt).__name__}")
    print(f"  tasks_config    : {type(tasks_config).__name__}, len={len(tasks_config)}")
    task0 = tasks_config[0]
    print(f"  task[0] keys    : {list(task0.keys())}")
    has_refs = "element_references" in task0 and task0["element_references"] is not None
    print(f"  element_references present: {has_refs}")
    if not has_refs:
        raise KeyError(
            f"task[0] in checkpoint '{ckpt_path}' has no element_references — "
            "cannot convert atomization energies to checkpoint reference frame"
        )
    refs_list = task0["element_references"]["element_references"]["_args_"][0]
    refs = torch.tensor(refs_list, dtype=torch.float64)  # [max_Z+1]
    print(f"  refs tensor     : shape={refs.shape}, H={refs[1]:.4f} eV, C={refs[6]:.4f} eV")
    return refs


# AIMNet2 reference energies (Ha → eV), same as process_aimnet2.py
_HA_TO_EV = 27.211386
_OUR_REFS_HA = {
    1:  -0.500607,    # H
    6:  -37.846772,   # C
    7:  -54.583861,   # N
    8:  -75.064579,   # O
    9:  -99.718730,   # F
    14: -289.359782,  # Si
    15: -341.259942,  # P
    16: -397.897380,  # S
    17: -460.117861,  # Cl
    35: -2573.966,    # Br
}
_OUR_REFS_EV: dict[int, float] | None = None


def get_our_refs_ev() -> dict[int, float]:
    global _OUR_REFS_EV
    if _OUR_REFS_EV is None:
        _OUR_REFS_EV = {z: e * _HA_TO_EV for z, e in _OUR_REFS_HA.items()}
    return _OUR_REFS_EV


def atomization_to_ckpt_frame(
    E_atomization: torch.Tensor,   # (n_graphs,) float32/64 — our atomization energies
    Z_list: list[torch.Tensor],    # list of (n_atoms_i,) tensors per graph
    ckpt_refs: torch.Tensor,       # [max_Z+1] float64
) -> torch.Tensor:
    """
    Convert our atomization energies to the checkpoint's reference frame.

    E_ckpt_target = E_atomization + sum_Z our_refs[Z] - sum_Z ckpt_refs[Z]
                  = E_dft        - sum_Z ckpt_refs[Z]
    """
    our_refs = get_our_refs_ev()
    corrections = []
    for Z in Z_list:
        our_sum  = sum(our_refs.get(int(z), 0.0) for z in Z.tolist())
        ckpt_sum = ckpt_refs[Z].sum().item()
        corrections.append(our_sum - ckpt_sum)
    corr = torch.tensor(corrections, dtype=torch.float64)
    return E_atomization.float() + corr.float()


# ── Dataset ───────────────────────────────────────────────────────────────────

class AseDbDataset(Dataset):
    """Loads an ASE SQLite DB and keeps each row as raw tensors.

    max_frames caps how many rows are loaded (after the max_atoms filter),
    so large DBs can be subsetted without reading the whole file.
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
    ), list(Zs)


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

def loss_fn(
    pred: dict,
    data: AtomicData,
    Z_list: list[torch.Tensor],
    ckpt_refs: torch.Tensor,
    device: str,
) -> torch.Tensor:
    raw_F  = pred["forces"] if torch.is_tensor(pred["forces"]) else pred["forces"]["forces"]
    pred_F = raw_F.float()

    force_loss = nn.functional.l1_loss(pred_F, data.forces.to(device))
    total      = FORCE_COEF * force_loss

    if ENERGY_COEF > 0.0:
        raw_E  = pred["energy"] if torch.is_tensor(pred["energy"]) else pred["energy"]["energy"]
        pred_E = raw_E.squeeze(-1).float()
        natoms = data.natoms.float().to(device)
        tgt_E  = atomization_to_ckpt_frame(data.energy.cpu(), Z_list, ckpt_refs).to(device)
        energy_loss = nn.functional.l1_loss(pred_E / natoms, tgt_E / natoms)
        total = total + ENERGY_COEF * energy_loss

    return total


# ── Train / Eval ──────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, device, ckpt_refs, train: bool):
    model.train(train)
    total_loss = total_fmae = total_steps = 0

    # Conservative forces require autograd through pos even during validation.
    with torch.enable_grad():
        for data, Z_list in loader:
            data = move_to(data, device)
            data.pos.requires_grad_(True)

            if train:
                optimizer.zero_grad()

            out = model(data)
            loss = loss_fn(out, data, Z_list, ckpt_refs, device)

            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 100.0)
                optimizer.step()

            raw_F = out["forces"] if torch.is_tensor(out["forces"]) else out["forces"]["forces"]
            f_mae = (raw_F.detach().float() - data.forces).abs().mean().item()
            total_loss  += loss.item()
            total_fmae  += f_mae
            total_steps += 1

    return total_loss / total_steps, total_fmae / total_steps


# ── EFPNorm swap ──────────────────────────────────────────────────────────────

def replace_norms_with_efp(model: nn.Module) -> int:
    """Replace all EquivariantRMSNorm layers in-place with EquivariantEFPNorm.

    Copies pretrained affine_weight / affine_bias so backbone output is identical
    at init (only the inverse-scale formula differs). Returns layers replaced.
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


# ── Main ──────────────────────────────────────────────────────────────────────

def build_datasets(args) -> tuple[list, list]:
    names = DATASETS if args.dataset == "all" else [args.dataset]
    train_ds_list, val_ds_list = [], []
    for name in names:
        tr_db = DATA_DIR / f"{name}_train.db"
        va_db = DATA_DIR / f"{name}_val.db"
        if not tr_db.exists():
            print(f"[{name}] SKIP — {tr_db} not found")
            continue
        train_ds_list.append(AseDbDataset(tr_db, CUTOFF, max_atoms=args.max_atoms,
                                          max_frames=args.max_train_frames))
        val_ds_list.append(AseDbDataset(va_db, CUTOFF, max_atoms=args.max_atoms,
                                        max_frames=args.max_val_frames))
    return train_ds_list, val_ds_list


class ConcatDataset(torch.utils.data.ConcatDataset):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS + ["all"], default="ani2x")
    parser.add_argument("--epochs",       type=int,   default=EPOCHS)
    parser.add_argument("--batch_size",   type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",           type=float, default=LR)
    parser.add_argument("--freeze_layers", type=int,  default=0,
                        help="Freeze first N backbone message-passing layers (0 = none)")
    parser.add_argument("--max_atoms",    type=int,   default=MAX_ATOMS)
    parser.add_argument("--resume",       type=str,   default=None,
                        help="Path to a previous finetune checkpoint to resume from")
    parser.add_argument("--out",          type=str,   default=None,
                        help="Output checkpoint path (default: scratch/finetune_<dataset>.pt)")
    parser.add_argument("--no_efp_norm",  action="store_true",
                        help="Disable EFPNorm replacement (use original RMSNorm)")
    parser.add_argument("--max_train_frames", type=int, default=None,
                        help="Cap train frames loaded per dataset (default: all)")
    parser.add_argument("--max_val_frames",   type=int, default=None,
                        help="Cap val frames loaded per dataset (default: all)")
    args = parser.parse_args()

    out_ckpt = Path(args.out) if args.out else _ROOT / "scratch" / f"finetune_{args.dataset}.pt"

    print(f"Device: {DEVICE}")
    print(f"Loading element references from checkpoint ...")
    ckpt_refs = load_element_refs(CKPT_PATH)
    print(f"  refs shape: {ckpt_refs.shape}  (H={ckpt_refs[1]:.2f} eV, C={ckpt_refs[6]:.2f} eV)")

    print(f"\nLoading datasets ...")
    train_ds_list, val_ds_list = build_datasets(args)
    if not train_ds_list:
        raise RuntimeError("No datasets found — run h5_to_asedb.py first")

    train_ds = ConcatDataset(train_ds_list) if len(train_ds_list) > 1 else train_ds_list[0]
    val_ds   = ConcatDataset(val_ds_list)   if len(val_ds_list)   > 1 else val_ds_list[0]

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    print(f"\nLoading pretrained eSEN checkpoint ...")
    model = initialize_finetuning_model(str(CKPT_PATH))

    if not args.no_efp_norm:
        n_replaced = replace_norms_with_efp(model)
        print(f"  EFPNorm: replaced {n_replaced} EquivariantRMSNorm layers")

    model = model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params/1e6:.1f}M parameters")

    if args.freeze_layers > 0:
        frozen = 0
        for name, param in model.backbone.named_parameters():
            if name.startswith(tuple(f"layers.{i}." for i in range(args.freeze_layers))):
                param.requires_grad_(False)
                frozen += param.numel()
        print(f"  Frozen {frozen/1e6:.1f}M params (first {args.freeze_layers} backbone layers)")

    start_epoch = 1
    if args.resume:
        state = torch.load(args.resume, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state["model"])
        start_epoch = state["epoch"] + 1
        print(f"  Resumed from epoch {state['epoch']}: {args.resume}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    for _ in range(start_epoch - 1):
        scheduler.step()

    print(f"\nFine-tuning for {args.epochs} epochs (LR={args.lr:.2e}) ...")
    best_val_fmae = float("inf")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_fmae = run_epoch(model, train_loader, optimizer, DEVICE, ckpt_refs, train=True)
        va_loss, va_fmae = run_epoch(model, val_loader,   optimizer, DEVICE, ckpt_refs, train=False)
        scheduler.step()
        dt = time.time() - t0

        star = " *" if va_fmae < best_val_fmae else ""
        if va_fmae < best_val_fmae:
            best_val_fmae = va_fmae
            torch.save({"model": model.state_dict(), "epoch": epoch}, out_ckpt)

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
