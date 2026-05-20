"""
NVE molecular dynamics evaluation for trained eSEN checkpoints.

Compares energy-conservation stability between EFPNorm and RMSNorm models.
Force MAE measures training accuracy; energy drift measures MD stability.

HOW THIS WORKS
--------------
1. ASE Calculator (eSENCalculator)
   ASE calls calculate(atoms) whenever it needs energy/forces — once per MD
   step (twice internally for Velocity Verlet, but both use the same wrapper).
   We convert the Atoms object to an AtomicData, rebuild the radius graph for
   the current geometry, and run the PyTorch forward pass.  Results (energy,
   forces) are stored in self.results; ASE reads them back via get_energy() /
   get_forces().

2. Conservative force path
   The backbone uses direct_forces=False, so forces = −∇_pos E via autograd.
   torch.enable_grad() MUST be active even at eval time (MLP_EFS_Head calls
   torch.autograd.grad internally), and pos.requires_grad_(True) must be set
   BEFORE the forward call.  Missing either causes zero/wrong forces.

3. Energy drift metric
   energy_drift_per_atom = max_t |E_total(t) − E_total(0)| / N_atoms
   Measures NVE conservation quality.  Good MLFF at 0.5 fs: ≲10 meV/atom.
   EFPNorm hypothesis: smaller drift because the full-rank Jacobian keeps
   force gradients accurate during training → smoother PES.

4. Failure modes to watch
   - NaN energy/forces: gradient instability in the norm layers, especially
     with RMSNorm at near-zero activations (1/eps blow-up in backward pass).
   - Force explosion (max_F > 500 eV/Å): geometry entered a steep repulsive
     wall; timestep too large or PES has a discontinuity.
   - Temperature blow-up (T > 10×T₀): energy is being injected — model is
     non-conservative or timestep is too large.
   - Monotonic energy drift: forces are not the true gradient of the predicted
     energy (should not happen with direct_forces=False, but worth checking).

Usage
-----
    python eval/run_md.py --checkpoint_dir train/checkpoints/qdpi_L4C128_efpnorm_lr4e-4
    python eval/run_md.py --checkpoint_dir train/checkpoints/qdpi_L4C128_rmsnorm_lr4e-4

Comparison workflow
-------------------
Run both commands above, then compare summary.json files:
    efpnorm : eval/md_runs/qdpi_L4C128_efpnorm_lr4e-4_val_s0_T300_dt0.5_N2000/summary.json
    rmsnorm : eval/md_runs/qdpi_L4C128_rmsnorm_lr4e-4_val_s0_T300_dt0.5_N2000/summary.json
"""

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from ase.db import connect
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from torch_cluster import radius_graph

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from fairchem.core.models.uma.escn_md import eSCNMDBackbone, MLP_EFS_Head
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.models.uma.nn.layer_norm import EquivariantRMSNormArraySphericalHarmonicsV2
from model.efpnorm import EquivariantEFPNorm

warnings.filterwarnings("ignore")

DATA_DIR = _ROOT / "data" / "asedb"

# Abort thresholds for failure detection
_MAX_FORCE_THRESHOLD = 500.0   # eV/Å
_TEMP_BLOW_UP_FACTOR = 10.0    # T > factor × T_init → unstable


# ── Model construction (mirrors train/pretrain.py exactly) ─────────────────────

class eSEN(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.backbone = eSCNMDBackbone(**cfg)
        self.head = MLP_EFS_Head(self.backbone, wrap_property=False)

    def forward(self, data: AtomicData) -> dict:
        return self.head(data, self.backbone(data))


def replace_norms_with_efp(model: nn.Module) -> int:
    """Swap all EquivariantRMSNorm layers for EquivariantEFPNorm in-place."""
    names = [
        name for name, mod in model.named_modules()
        if isinstance(mod, EquivariantRMSNormArraySphericalHarmonicsV2)
    ]
    for name in names:
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        old = getattr(parent, parts[-1])
        new = EquivariantEFPNorm(
            lmax=old.lmax,
            num_channels=old.num_channels,
            affine=old.affine,
            normalization=old.normalization,
            centering=old.centering,
            std_balance_degrees=old.std_balance_degrees,
        )
        if old.affine_weight is not None:
            new.affine_weight.data.copy_(old.affine_weight.data)
        if old.affine_bias is not None:
            new.affine_bias.data.copy_(old.affine_bias.data)
        setattr(parent, parts[-1], new)
    return len(names)


def load_checkpoint(checkpoint_dir: Path, ckpt_file: str, device: str):
    """Load config.json + best.pt, return (model, config dict)."""
    with open(checkpoint_dir / "config.json") as f:
        config = json.load(f)

    backbone_cfg = config["backbone_cfg"]
    efp_norm     = config.get("efp_norm", False)

    model = eSEN(backbone_cfg)

    if efp_norm:
        n = replace_norms_with_efp(model)
        print(f"  EFPNorm: replaced {n} EquivariantRMSNorm layers")
    else:
        print("  Using original EquivariantRMSNorm (RMSNorm baseline)")

    state = torch.load(str(checkpoint_dir / ckpt_file), map_location=device,
                       weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params / 1e6:.1f}M parameters  device={device}")
    return model, config


# ── AtomicData helpers ─────────────────────────────────────────────────────────

def _move_to(data: AtomicData, device: str) -> AtomicData:
    for k in ("pos", "atomic_numbers", "cell", "pbc", "natoms",
              "edge_index", "cell_offsets", "nedges",
              "charge", "spin", "fixed", "tags", "energy", "forces", "batch"):
        v = getattr(data, k, None)
        if v is not None:
            setattr(data, k, v.to(device))
    return data


def atoms_to_atomicdata(atoms, cutoff: float) -> AtomicData:
    """Convert one ASE Atoms → AtomicData for a single-molecule forward pass.

    Rebuilds the radius graph from scratch each call because atom positions
    change every MD step, so the neighbor list is always stale.
    """
    n = len(atoms)
    Z   = torch.tensor(atoms.numbers.copy(), dtype=torch.long)
    pos = torch.tensor(atoms.positions.copy(), dtype=torch.float32)
    bat = torch.zeros(n, dtype=torch.long)   # all atoms in graph 0

    edge_index = radius_graph(pos, r=cutoff, batch=bat, loop=False)
    n_edges    = edge_index.shape[1]

    return AtomicData(
        pos            = pos,
        atomic_numbers = Z,
        cell           = torch.zeros(1, 3, 3),
        pbc            = torch.zeros(1, 3, dtype=torch.bool),
        natoms         = torch.tensor([n]),
        edge_index     = edge_index,
        cell_offsets   = torch.zeros(n_edges, 3),
        nedges         = torch.tensor([n_edges]),
        charge         = torch.zeros(1, dtype=torch.long),
        spin           = torch.ones(1,  dtype=torch.long),
        fixed          = torch.zeros(n, dtype=torch.long),
        tags           = torch.zeros(n, dtype=torch.long),
        energy         = torch.zeros(1),
        forces         = torch.zeros(n, 3),
        batch          = bat,
        sid            = ["0"],
        dataset        = None,
    )


def _extract_energy(out: dict) -> float:
    e = out["energy"]
    t = e if torch.is_tensor(e) else e["energy"]
    return float(t.reshape(-1)[0].item())


def _extract_forces(out: dict) -> torch.Tensor:
    f = out["forces"]
    return f if torch.is_tensor(f) else f["forces"]


# ── ASE Calculator ─────────────────────────────────────────────────────────────

class eSENCalculator(Calculator):
    """ASE Calculator that wraps a trained eSEN PyTorch model.

    Each call to calculate():
      1. Copies current atom positions/numbers to float32 tensors
      2. Builds a radius graph (neighbor list) for the current geometry
      3. Packs everything into an AtomicData object and moves to device
      4. Sets pos.requires_grad_(True) so autograd can compute −∇_pos E
      5. Runs the forward pass under torch.enable_grad()
         (required even at eval time: MLP_EFS_Head calls autograd.grad internally)
      6. Stores energy (float, eV) and forces (ndarray N×3, eV/Å) in self.results

    ASE reads self.results['energy'] / self.results['forces'] via get_energy() /
    get_forces(), and the VelocityVerlet integrator uses forces each step.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, model: eSEN, cutoff: float, device: str, **kwargs):
        super().__init__(**kwargs)
        self.model   = model
        self.cutoff  = cutoff
        self.device  = device
        self.n_calls = 0

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        # super() stores atoms as self.atoms and validates system_changes
        super().calculate(atoms, properties, system_changes)

        data = atoms_to_atomicdata(self.atoms, self.cutoff)
        data = _move_to(data, self.device)
        data.pos.requires_grad_(True)   # required for conservative forces

        with torch.enable_grad():       # must be on even in model.eval() mode
            out = self.model(data)

        energy = _extract_energy(out)
        forces = _extract_forces(out).detach().float().cpu().numpy().astype(np.float64)

        self.results["energy"] = energy   # float, eV
        self.results["forces"] = forces   # (N, 3) float64, eV/Å
        self.n_calls += 1


# ── Simulation failure ─────────────────────────────────────────────────────────

class SimulationFailed(Exception):
    pass


# ── Main NVE routine ──────────────────────────────────────────────────────────

def run_md(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint_dir = Path(args.checkpoint_dir)
    run_name = checkpoint_dir.name

    # ── Output directory ──────────────────────────────────────────────────────
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        tag = (f"{run_name}_{args.split}_s{args.sample_idx}"
               f"_T{int(args.temperature)}_dt{args.timestep_fs}_N{args.steps}")
        out_dir = _ROOT / "eval" / "md_runs" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*64}")
    print(f"  MD run : {run_name}")
    print(f"  Output : {out_dir}")
    print(f"{'='*64}")

    # ── Device ────────────────────────────────────────────────────────────────
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("  WARNING: CUDA not available, falling back to CPU")
        device = "cpu"

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\n[1] Loading checkpoint: {checkpoint_dir.name} / {args.ckpt_file}")
    model, config = load_checkpoint(checkpoint_dir, args.ckpt_file, device)
    cutoff = config["backbone_cfg"]["cutoff"]

    # ── Load molecule ─────────────────────────────────────────────────────────
    db_path = DATA_DIR / f"{args.dataset}_{args.split}.db"
    print(f"\n[2] Loading molecule idx={args.sample_idx} from {db_path.name}")

    with connect(str(db_path)) as db:
        rows = list(db.select(limit=args.sample_idx + 1))
    if len(rows) <= args.sample_idx:
        raise ValueError(
            f"sample_idx={args.sample_idx} out of range "
            f"(DB has only {len(rows)} rows with limit)"
        )
    atoms = rows[args.sample_idx].toatoms()
    n_atoms  = len(atoms)
    formula  = atoms.get_chemical_formula()
    print(f"  Formula: {formula}  N_atoms: {n_atoms}")

    # ── Attach calculator ─────────────────────────────────────────────────────
    print(f"\n[3] Attaching eSENCalculator  cutoff={cutoff} Å")
    calc = eSENCalculator(model=model, cutoff=cutoff, device=device)
    atoms.calc = calc

    # ── Initialize velocities ─────────────────────────────────────────────────
    print(f"\n[4] Initializing velocities: Maxwell-Boltzmann @ {args.temperature} K")
    np.random.seed(args.seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)
    T_init = atoms.get_temperature()
    print(f"  Initial temperature: {T_init:.1f} K")

    # ── Record step 0 ─────────────────────────────────────────────────────────
    print(f"\n[5] Running NVE: {args.steps} steps × {args.timestep_fs} fs "
          f"= {args.steps * args.timestep_fs / 1000:.2f} ps")

    metrics      = []
    failed       = False
    fail_reason  = ""
    steps_done   = 0

    # Trigger initial force/energy evaluation at step 0
    epot0 = atoms.get_potential_energy()
    ekin0 = atoms.get_kinetic_energy()
    etot0 = epot0 + ekin0

    if np.isnan(epot0):
        failed      = True
        fail_reason = "NaN energy at step 0"
        print(f"  ERROR: {fail_reason}")
    else:
        forces0   = calc.results["forces"]    # cached, no extra model call
        max_f0    = float(np.max(np.linalg.norm(forces0, axis=1)))
        metrics.append({
            "step": 0, "epot": epot0, "ekin": ekin0, "etot": etot0,
            "temp": T_init, "max_force": max_f0, "drift_per_atom": 0.0, "nan": 0,
        })
        print(f"  step      0  T={T_init:6.1f} K  Etot={etot0:.5f} eV  "
              f"drift=0.000 meV/atom  Fmax={max_f0:.2f} eV/Å")

    # ── Dynamics ──────────────────────────────────────────────────────────────
    if not failed:
        timestep  = args.timestep_fs * units.fs
        traj_path = out_dir / "trajectory.traj"

        # dyn is captured by the closure below once assigned
        dyn = VelocityVerlet(atoms, timestep=timestep, trajectory=str(traj_path))

        def record_and_check():
            """Called by ASE every record_interval steps."""
            s = dyn.nsteps
            # ASE fires observers at nsteps=0 before any steps run; skip it
            # because we already recorded step 0 manually above.
            if s == 0:
                return
            epot      = calc.results["energy"]          # cached from last step
            forces_np = calc.results["forces"]          # cached from last step
            ekin      = atoms.get_kinetic_energy()      # from velocities, no model call
            temp      = atoms.get_temperature()         # from velocities, no model call
            etot      = epot + ekin
            max_force = float(np.max(np.linalg.norm(forces_np, axis=1)))
            drift     = abs(etot - etot0) / n_atoms

            nan_hit = (np.isnan(epot) or np.isnan(ekin)
                       or np.isnan(forces_np).any())

            metrics.append({
                "step": s, "epot": epot, "ekin": ekin, "etot": etot,
                "temp": temp, "max_force": max_force,
                "drift_per_atom": drift, "nan": int(nan_hit),
            })

            if s % 200 == 0:
                print(f"  step {s:5d}  T={temp:6.1f} K  Etot={etot:.5f} eV  "
                      f"drift={drift*1000:6.2f} meV/atom  Fmax={max_force:.2f} eV/Å")

            # Abort conditions
            if nan_hit:
                raise SimulationFailed(f"NaN at step {s}")
            if max_force > _MAX_FORCE_THRESHOLD:
                raise SimulationFailed(
                    f"Force explosion at step {s}: {max_force:.1f} eV/Å"
                )
            if temp > _TEMP_BLOW_UP_FACTOR * args.temperature:
                raise SimulationFailed(
                    f"Temperature blow-up at step {s}: {temp:.1f} K "
                    f"(>{_TEMP_BLOW_UP_FACTOR}×{args.temperature} K)"
                )

        dyn.attach(record_and_check, interval=args.record_interval)

        try:
            dyn.run(args.steps)
            steps_done = args.steps
        except SimulationFailed as exc:
            failed      = True
            fail_reason = str(exc)
            steps_done  = dyn.nsteps
            print(f"\n  SIMULATION FAILED: {fail_reason}")

        print(f"\n  Completed {steps_done}/{args.steps} steps  "
              f"({calc.n_calls} model calls)")

    # ── Compute summary metrics ────────────────────────────────────────────────
    if metrics:
        etot_arr = np.array([m["etot"] for m in metrics])
        temp_arr = np.array([m["temp"] for m in metrics])
        drift_pa = float(np.max(np.abs(etot_arr - etot_arr[0]))) / n_atoms
        max_temp = float(np.max(temp_arr))
        print(f"\n[6] Energy conservation")
        print(f"  energy_drift_per_atom = {drift_pa*1000:.3f} meV/atom")
        print(f"  max_temperature       = {max_temp:.1f} K")
    else:
        drift_pa = float("nan")
        max_temp = float("nan")

    # ── Save md_metrics.csv ───────────────────────────────────────────────────
    if metrics:
        csv_path = out_dir / "md_metrics.csv"
        cols = ["step", "epot", "ekin", "etot", "temp", "max_force",
                "drift_per_atom", "nan"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(metrics)
        print(f"\n  Saved: {csv_path}")
    if (out_dir / "trajectory.traj").exists():
        print(f"  Saved: {out_dir / 'trajectory.traj'}")

    # ── Save summary.json ─────────────────────────────────────────────────────
    summary = {
        "run_name":                   run_name,
        "checkpoint_dir":             str(checkpoint_dir),
        "ckpt_file":                  args.ckpt_file,
        "dataset":                    args.dataset,
        "split":                      args.split,
        "sample_idx":                 args.sample_idx,
        "formula":                    formula,
        "n_atoms":                    n_atoms,
        "temperature_init_K":         round(T_init, 2) if not failed or metrics else None,
        "temperature_requested_K":    args.temperature,
        "timestep_fs":                args.timestep_fs,
        "steps_requested":            args.steps,
        "steps_completed":            steps_done,
        "record_interval":            args.record_interval,
        "efp_norm":                   config.get("efp_norm", False),
        "device":                     device,
        "seed":                       args.seed,
        "failed":                     failed,
        "fail_reason":                fail_reason,
        "energy_drift_per_atom_eV":   None if np.isnan(drift_pa) else drift_pa,
        "energy_drift_per_atom_meV":  None if np.isnan(drift_pa) else round(drift_pa * 1000, 4),
        "max_temperature_K":          None if np.isnan(max_temp) else round(max_temp, 2),
        "n_model_calls":              calc.n_calls,
        "initial_Etot_eV":            float(etot0) if not failed else None,
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")

    # ── Final result ──────────────────────────────────────────────────────────
    status = "FAILED" if failed else "STABLE"
    print(f"\n{'='*64}")
    if failed:
        print(f"  STATUS : {status}  ({fail_reason})")
        print(f"  Completed {steps_done}/{args.steps} steps before failure")
    else:
        print(f"  STATUS : {status}")
        print(f"  energy_drift_per_atom = {drift_pa*1000:.3f} meV/atom")
    print(f"{'='*64}\n")

    return summary


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NVE MD evaluation for trained eSEN checkpoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint_dir", required=True,
        help="Checkpoint directory, e.g. train/checkpoints/qdpi_L4C128_efpnorm_lr4e-4",
    )
    parser.add_argument(
        "--ckpt_file", default="best.pt",
        help="Checkpoint filename inside checkpoint_dir",
    )
    parser.add_argument(
        "--dataset", default="qdpi",
        choices=["aimnet2", "spice2", "qdpi", "ani2x", "spf"],
    )
    parser.add_argument(
        "--split", default="val", choices=["train", "val"],
    )
    parser.add_argument(
        "--sample_idx", type=int, default=0,
        help="0-based row index in the DB split",
    )
    parser.add_argument(
        "--temperature", type=float, default=300.0,
        help="Initial temperature for Maxwell-Boltzmann velocity initialization (K)",
    )
    parser.add_argument(
        "--timestep_fs", type=float, default=0.5,
        help="MD timestep (fs)",
    )
    parser.add_argument(
        "--steps", type=int, default=2000,
        help="Total NVE steps (2000 × 0.5 fs = 1 ps)",
    )
    parser.add_argument(
        "--record_interval", type=int, default=10,
        help="Record metrics every N steps",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="PyTorch device (falls back to cpu if cuda is unavailable)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--out_dir", default=None,
        help="Output directory (default: eval/md_runs/<auto-name>)",
    )
    args = parser.parse_args()

    # Resolve relative checkpoint_dir against project root
    ckpt = Path(args.checkpoint_dir)
    if not ckpt.is_absolute():
        ckpt = _ROOT / ckpt
    args.checkpoint_dir = ckpt

    run_md(args)


if __name__ == "__main__":
    main()
