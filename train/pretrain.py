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
import csv
import json
import logging
import re as _re
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

from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeElapsedColumn, MofNCompleteColumn,
)
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

console = Console(highlight=False)

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
    cs_emb_grad         = True,        # official eSEN-SM value (NOT False)
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
MAX_ATOMS  = 100
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

def _row_to_tensors(row):
    atoms = row.toatoms()
    return (
        torch.tensor(atoms.numbers.copy(),                          dtype=torch.long),
        torch.tensor(atoms.positions.copy().astype(np.float32),     dtype=torch.float32),
        torch.tensor(float(atoms.get_potential_energy()),           dtype=torch.float32),
        torch.tensor(atoms.get_forces().copy().astype(np.float32),  dtype=torch.float32),
    )


class AseDbDataset(Dataset):
    """Lazy-loading ASE SQLite dataset.

    __init__ builds (or reloads from disk) a filtered index of row IDs by
    scanning row.natoms — it never calls toatoms(), so startup is fast even
    for tens of millions of frames.  The index is cached alongside the DB
    as <stem>.idx_ma<max_atoms>.npy and invalidated by DB mtime.

    Actual atom data is fetched on demand in __getitem__ via a per-worker
    SQLite connection opened lazily on first access.  Each DataLoader worker
    process gets its own independent handle, so concurrent reads are safe.

    Set preload=True only for tiny smoke-test datasets; for large-scale
    training the lazy path keeps RAM flat regardless of dataset size.
    """
    def __init__(self, db_path: Path, cutoff: float, max_atoms: int = MAX_ATOMS,
                 max_frames: int | None = None, preload: bool = False):
        self.db_path   = db_path
        self.cutoff    = cutoff
        self.max_atoms = max_atoms
        self.preload   = preload
        self._conn     = None  # opened lazily; None prevents inheriting across fork

        cache_path = db_path.parent / f"{db_path.stem}.idx_ma{max_atoms}.npy"
        if cache_path.exists() and cache_path.stat().st_mtime >= db_path.stat().st_mtime:
            row_ids = np.load(cache_path).tolist()
            console.print(f"  [dim]cached index[/]  {db_path.name}  "
                          f"[dim]({len(row_ids):,} rows before cap)[/]")
        else:
            row_ids = []
            with connect(str(db_path)) as db:
                total = db.count()
                with Progress(
                    SpinnerColumn(),
                    TextColumn(f"  [cyan]indexing[/]  {db_path.name}"),
                    BarColumn(bar_width=28),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                    console=console, transient=True,
                ) as prog:
                    task = prog.add_task("", total=total)
                    for row in db.select():
                        if row.natoms <= max_atoms:
                            row_ids.append(row.id)
                        prog.advance(task)
            np.save(cache_path, np.array(row_ids, dtype=np.int64))

        if max_frames is not None:
            row_ids = row_ids[:max_frames]
        self.row_ids = row_ids

        cap = f"/{max_frames:,}" if max_frames is not None else ""
        console.print(f"  [green]✓[/] [bold]{db_path.name}[/]  "
                      f"[cyan]{len(self.row_ids):,}[/][dim]{cap} frames[/]")

        if preload:
            conn = connect(str(db_path))
            with Progress(
                SpinnerColumn(),
                TextColumn(f"  [yellow]preloading[/]  {db_path.name}"),
                BarColumn(bar_width=28),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console, transient=True,
            ) as prog:
                task = prog.add_task("", total=len(self.row_ids))
                self._data = []
                for rid in self.row_ids:
                    self._data.append(_row_to_tensors(conn.get(id=rid)))
                    prog.advance(task)
            conn.close()

    def __getstate__(self):
        # Never pickle an open connection — each worker must open its own.
        state = self.__dict__.copy()
        state["_conn"] = None
        return state

    def __len__(self):
        return len(self.row_ids)

    def __getitem__(self, idx):
        if self.preload:
            return self._data[idx]
        if self._conn is None:
            self._conn = connect(str(self.db_path))
        return _row_to_tensors(self._conn.get(id=self.row_ids[idx]))


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

def run_epoch(model, loader, optimizer, device, train: bool,
              progress=None, task_id=None):
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
            if progress is not None:
                progress.advance(task_id)

    return total_loss / total_steps, total_fmae / total_steps


# ── Experiment logging ────────────────────────────────────────────────────────

def _make_run_name(args) -> str:
    mantissa, exp = f"{args.lr:.0e}".split("e")
    lr_str = f"{mantissa}e{int(exp)}"
    norm   = "rmsnorm" if args.no_efp_norm else "efpnorm"
    return f"{args.dataset}_L{args.num_layers}C{args.sphere_channels}_{norm}_lr{lr_str}"


def _setup_file_log(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("pretrain")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path, mode="a")
        fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s",
                                          datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(fh)
    return logger


def _save_config(ckpt_dir: Path, args, cfg: dict, run_name: str) -> None:
    config = {
        "run_name":         run_name,
        "dataset":          args.dataset,
        "lr":               args.lr,
        "epochs":           args.epochs,
        "batch_size":       args.batch_size,
        "num_layers":       args.num_layers,
        "sphere_channels":  args.sphere_channels,
        "hidden_channels":  cfg["hidden_channels"],
        "edge_channels":    cfg["edge_channels"],
        "efp_norm":         not args.no_efp_norm,
        "max_train_frames": args.max_train_frames,
        "max_val_frames":   args.max_val_frames,
        "num_workers":      args.num_workers,
        "seed":             SEED,
        "backbone_cfg":     dict(cfg),
    }
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


_METRICS_COLS = ["epoch", "train_loss", "train_fmae", "val_loss", "val_fmae",
                 "lr", "epoch_time_sec", "best_val_fmae"]


def _append_metrics(ckpt_dir: Path, row: dict) -> None:
    path   = ckpt_dir / "metrics.csv"
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_METRICS_COLS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def _save_summary(ckpt_dir: Path, summary: dict) -> None:
    with open(ckpt_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


# ── Main ──────────────────────────────────────────────────────────────────────

def build_datasets(args):
    names = DATASETS if args.dataset == "all" else [args.dataset]
    train_list, val_list = [], []
    for name in names:
        tr_db = DATA_DIR / f"{name}_train.db"
        va_db = DATA_DIR / f"{name}_val.db"
        if not tr_db.exists() or not va_db.exists():
            _skip_dataset(name, tr_db if not tr_db.exists() else va_db)
            continue
        train_list.append(AseDbDataset(tr_db, CUTOFF, max_atoms=args.max_atoms,
                                       max_frames=args.max_train_frames,
                                       preload=args.preload_data))
        val_list.append(AseDbDataset(va_db, CUTOFF, max_atoms=args.max_atoms,
                                     max_frames=args.max_val_frames,
                                     preload=args.preload_data))
    return train_list, val_list


def _skip_dataset(name: str, path: Path) -> None:
    console.print(f"  [dim][{name}] SKIP — {path} not found[/]")


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
                        help="Resume from checkpoint dir (uses latest.pt) or a specific .pt file")
    parser.add_argument("--out",          type=str,   default=None,
                        help="Checkpoint directory (default: train/checkpoints/pretrain_<dataset>)")
    parser.add_argument("--no_efp_norm",  action="store_true",
                        help="Disable EFPNorm (use original EquivariantRMSNorm)")
    parser.add_argument("--max_train_frames", type=int, default=None)
    parser.add_argument("--max_val_frames",   type=int, default=None)
    parser.add_argument("--preload_data",  action="store_true",
                        help="Preload all frames into RAM (only for small smoke tests)")
    parser.add_argument("--num_workers",   type=int,   default=4,
                        help="DataLoader worker processes (0 = main process only)")
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

    # ── Run directory ─────────────────────────────────────────────────────────
    run_name = Path(args.out).name if args.out else _make_run_name(args)
    ckpt_dir = Path(args.out) if args.out else _ROOT / "train" / "checkpoints" / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt   = ckpt_dir / "best.pt"
    latest_ckpt = ckpt_dir / "latest.pt"

    logger = _setup_file_log(ckpt_dir / "stdout.log")
    logger.info(f"=== run={run_name} ===")

    # ── Load data ─────────────────────────────────────────────────────────────
    console.print(Rule("[dim]datasets[/]", style="dim"))
    train_list, val_list = build_datasets(args)
    if not train_list:
        raise RuntimeError("No datasets found")

    train_ds = ConcatDataset(train_list) if len(train_list) > 1 else train_list[0]
    val_ds   = ConcatDataset(val_list)   if len(val_list)   > 1 else val_list[0]

    nw = 0 if args.preload_data else args.num_workers
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=nw,
                              persistent_workers=nw > 0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=nw,
                              persistent_workers=nw > 0)

    # ── Build model ───────────────────────────────────────────────────────────
    console.print(Rule("[dim]model[/]", style="dim"))
    cfg = {
        **BACKBONE_CFG,
        "num_layers":      args.num_layers,
        "sphere_channels": args.sphere_channels,
        "hidden_channels": hidden_channels,
        "edge_channels":   edge_channels,
    }
    n_replaced = 0
    with console.status("[cyan]Building eSEN from scratch ...[/]", spinner="dots"):
        model = eSEN(cfg)
        if not args.no_efp_norm:
            n_replaced = replace_norms_with_efp(model)
        model = model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    norm_str = (f"[green]EFPNorm[/] [dim]×{n_replaced}[/]"
                if not args.no_efp_norm else "[dim]EquivariantRMSNorm[/]")
    console.print(f"  [green]✓[/] layers=[bold]{cfg['num_layers']}[/]  "
                  f"ch=[bold]{cfg['sphere_channels']}/{cfg['hidden_channels']}/{cfg['edge_channels']}[/]  "
                  f"[dim]{n_params/1e6:.1f}M params[/]  {norm_str}")

    # ── Save config ───────────────────────────────────────────────────────────
    _save_config(ckpt_dir, args, cfg, run_name)

    # ── Optimizer / scheduler ─────────────────────────────────────────────────
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

    start_epoch   = 1
    best_val_fmae = float("inf")
    best_epoch    = 0
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.is_dir():
            resume_path = resume_path / "latest.pt"
        state = torch.load(resume_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state["model"])
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        start_epoch   = state["epoch"] + 1
        best_val_fmae = state.get("best_val_fmae", float("inf"))
        best_epoch    = state.get("best_epoch", 0)
        # Advance the scheduler (T_max=args.epochs) to the correct position
        # in the new cosine schedule rather than restoring the old state.
        # Scheduler was constructed with base_lrs=[args.lr] before optimizer
        # state was loaded, so base_lrs are unaffected by the state restore.
        for _ in range(start_epoch - 1):
            scheduler.step()
        console.print(f"  [yellow]↑ resumed[/] epoch {state['epoch']}  "
                      f"best F-MAE {best_val_fmae*1000:.1f} meV/Å  "
                      f"lr={scheduler.get_last_lr()[0]:.2e}  "
                      f"[dim]{resume_path}[/]")
        logger.info(f"resumed from epoch {state['epoch']}  "
                    f"best_val_fmae={best_val_fmae:.6f}  path={resume_path}")

    # ── Summary panel ─────────────────────────────────────────────────────────
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", justify="right")
    grid.add_column()
    grid.add_row("run",          f"[bold]{run_name}[/]")
    grid.add_row("dataset",      f"[bold]{args.dataset}[/]  "
                                 f"[dim]{len(train_ds):,} train / {len(val_ds):,} val[/]")
    grid.add_row("architecture", f"layers={cfg['num_layers']}  "
                                 f"ch={cfg['sphere_channels']}/{cfg['hidden_channels']}/{cfg['edge_channels']}  "
                                 f"[dim]{n_params/1e6:.1f}M params[/]")
    grid.add_row("norm",         norm_str)
    grid.add_row("training",     f"epochs={args.epochs}  lr={args.lr:.2e}  "
                                 f"batch={args.batch_size}  workers={nw}")
    grid.add_row("device",       f"[bold]{DEVICE}[/]")
    grid.add_row("checkpoints",  f"[dim]{ckpt_dir}[/]")
    console.print(Panel(grid, title="[bold blue]eSEN Pretraining[/]",
                        border_style="blue", padding=(0, 1)))
    logger.info(f"dataset={args.dataset}  epochs={args.epochs}  lr={args.lr}  "
                f"batch={args.batch_size}  layers={cfg['num_layers']}  "
                f"ch={cfg['sphere_channels']}  efp_norm={not args.no_efp_norm}  "
                f"params={n_params/1e6:.1f}M  device={DEVICE}")

    # ── Training loop ─────────────────────────────────────────────────────────
    train_start = time.time()
    w = len(str(args.epochs))
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=28),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console, transient=True,
        ) as prog:
            tr_task = prog.add_task(
                f"[cyan]Epoch {epoch:{w}d}/{args.epochs} train[/]",
                total=len(train_loader))
            tr_loss, tr_fmae = run_epoch(
                model, train_loader, optimizer, DEVICE, train=True,
                progress=prog, task_id=tr_task)
            va_task = prog.add_task(
                f"[blue]Epoch {epoch:{w}d}/{args.epochs}   val[/]",
                total=len(val_loader))
            va_loss, va_fmae = run_epoch(
                model, val_loader, optimizer, DEVICE, train=False,
                progress=prog, task_id=va_task)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        dt = time.time() - t0

        improved = va_fmae < best_val_fmae
        if improved:
            best_val_fmae = va_fmae
            best_epoch    = epoch
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "backbone_cfg": cfg}, best_ckpt)

        # Full state every epoch so --resume restores optimizer/scheduler momentum.
        torch.save({
            "model":         model.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "scheduler":     scheduler.state_dict(),
            "epoch":         epoch,
            "best_val_fmae": best_val_fmae,
            "best_epoch":    best_epoch,
            "backbone_cfg":  cfg,
        }, latest_ckpt)

        _append_metrics(ckpt_dir, {
            "epoch":          epoch,
            "train_loss":     round(tr_loss,  6),
            "train_fmae":     round(tr_fmae,  6),
            "val_loss":       round(va_loss,  6),
            "val_fmae":       round(va_fmae,  6),
            "lr":             round(current_lr, 8),
            "epoch_time_sec": round(dt, 1),
            "best_val_fmae":  round(best_val_fmae, 6),
        })

        va_color = "bold green" if improved else "cyan"
        star     = "  [bold green]★ best[/]" if improved else ""
        console.print(
            f"[dim]Epoch[/] [bold]{epoch:{w}d}[/][dim]/{args.epochs}[/]  "
            f"[dim]train[/] {tr_loss:.4f} [yellow]{tr_fmae*1000:6.1f}[/][dim] meV/Å[/]  "
            f"[dim]│  val[/] {va_loss:.4f} [{va_color}]{va_fmae*1000:6.1f}[/{va_color}][dim] meV/Å[/]  "
            f"[dim]{dt:.0f}s[/]{star}"
        )
        logger.info(
            f"epoch={epoch}  "
            f"train_loss={tr_loss:.4f}  train_fmae={tr_fmae*1000:.1f}meV/A  "
            f"val_loss={va_loss:.4f}  val_fmae={va_fmae*1000:.1f}meV/A  "
            f"lr={current_lr:.2e}  time={dt:.0f}s"
            + ("  [best]" if improved else "")
        )

    # ── Final summary ─────────────────────────────────────────────────────────
    total_time = time.time() - train_start
    summary = {
        "run_name":                run_name,
        "best_val_fmae_mevA":      round(best_val_fmae * 1000, 3),
        "best_epoch":              best_epoch,
        "total_training_time_sec": round(total_time, 1),
        "total_epochs_completed":  args.epochs - start_epoch + 1,
        "checkpoint_dir":          str(ckpt_dir),
        "final_learning_rate":     scheduler.get_last_lr()[0],
    }
    _save_summary(ckpt_dir, summary)

    console.print(Rule(style="blue dim"))
    console.print(f"[bold green]Best val F-MAE: {best_val_fmae*1000:.1f} meV/Å  "
                  f"[dim](epoch {best_epoch})[/][/]")
    console.print(f"[dim]best.pt   →[/] {best_ckpt}")
    console.print(f"[dim]latest.pt →[/] {latest_ckpt}")
    logger.info(f"=== done  best_val_fmae={best_val_fmae*1000:.1f}meV/A  "
                f"best_epoch={best_epoch}  total_time={total_time:.0f}s ===")


if __name__ == "__main__":
    main()
