"""
Long-duration NVE MD pilot: efpnorm vs rmsnorm on a hand-picked set of 8
AIMNet2 val molecules (3 RMSNorm-worse outliers, 3 EFPNorm-worse outliers,
2 near-zero-gap "typical" molecules — all 31-58 atoms), at a duration well
beyond the standard 2.5 ps screening runs.

Goal: check whether the energy-drift gap between the two norm schemes grows
with trajectory length (real accumulating instability) or stays flat
(short-run noise). See eval/md_runs/comparison_aimnet2_full200.json for how
the molecule set was selected.

Standalone script — does not touch run_md_compare.py / run_md_compare_l6.py
or their existing outputs.

Usage (from efpnorm-esen root):
    python eval/run_md_longpilot.py --device0 cuda:0 --device1 cuda:3 --steps 40000
"""

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent
_PYTHON = "/nethome/kyang394/scratch/envs/MLFF/bin/python"
_RUN_MD = str(_ROOT / "eval" / "run_md.py")

CHECKPOINTS = {
    "efpnorm": str(_ROOT / "train/checkpoints/aimnet2_L4C128_efpnorm_lr4e-4"),
    "rmsnorm": str(_ROOT / "train/checkpoints/aimnet2_L4C128_rmsnorm_lr4e-4"),
}

# idx: (formula, category) — pulled from comparison_aimnet2_full200.json
MOLECULES = {
    39411:  "C17H25N3 (rms-worse, gap=240)",
    253132: "C15H22O (rms-worse, gap=29)",
    223835: "C9H20N4Si2 (rms-worse, gap=20)",
    357430: "C13H25NO5S (efp-worse, gap=-31)",
    244658: "C16H31N5O6 (efp-worse, gap=-22)",
    146045: "C14H28S (efp-worse, gap=-21)",
    414475: "C11H25N3OS (typical)",
    212470: "C12H16F2N4O2 (typical)",
}


def out_dir_name(label, idx, steps, timestep, temperature):
    run_name = f"aimnet2_L4C128_{label}_lr4e-4"
    return (f"{run_name}_val_s{idx}"
            f"_T{int(temperature)}_dt{timestep}_N{steps}_seed42")


def run_batch(label, ckpt, device, indices, steps, timestep, temperature, skip_existing=False):
    out_dirs = []
    for idx in indices:
        tag = out_dir_name(label, idx, steps, timestep, temperature)
        out_dir = _ROOT / "eval" / "md_runs" / tag
        out_dirs.append(out_dir)

        if skip_existing and (out_dir / "summary.json").exists():
            print(f"  [{label}] mol={idx} skipping (already done)", flush=True)
            continue

        cmd = [
            _PYTHON, _RUN_MD,
            "--checkpoint_dir", ckpt,
            "--dataset", "aimnet2",
            "--split", "val",
            "--sample_idx", str(idx),
            "--temperature", str(temperature),
            "--timestep_fs", str(timestep),
            "--steps", str(steps),
            "--record_interval", "200",
            "--device", device,
            "--seed", "42",
            "--out_dir", str(out_dir),
        ]
        print(f"  [{label}] mol={idx} ({MOLECULES.get(idx,'?')}) starting on {device}...", flush=True)
        t0 = time.time()
        result = subprocess.run(cmd, cwd=str(_ROOT), capture_output=True, text=True)
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f"  [{label}] mol={idx} ERROR after {elapsed:.0f}s")
            print(result.stderr[-1000:])
        else:
            summary_path = out_dir / "summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    s = json.load(f)
                drift = s.get("energy_drift_per_atom_meV")
                failed = s.get("failed", False)
                if failed:
                    print(f"  [{label}] mol={idx} FAILED ({s.get('fail_reason','?')}) after {elapsed:.0f}s", flush=True)
                else:
                    print(f"  [{label}] mol={idx} drift={drift:.3f} meV/atom elapsed={elapsed:.0f}s", flush=True)
    return out_dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device0", default="cuda:0", help="GPU for efpnorm")
    parser.add_argument("--device1", default="cuda:1", help="GPU for rmsnorm")
    parser.add_argument("--steps", type=int, default=40000, help="40000 steps x 0.5fs = 20ps")
    parser.add_argument("--timestep", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--sample_indices", type=int, nargs="+",
                        default=list(MOLECULES.keys()))
    parser.add_argument("--out_json", default=None)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    indices = args.sample_indices
    ps = args.steps * args.timestep / 1000

    print(f"\n{'='*70}")
    print(f"  LONG MD PILOT: efpnorm vs rmsnorm  (dataset=aimnet2, L4C128)")
    print(f"  Steps={args.steps} × dt={args.timestep} fs = {ps:.2f} ps  T={args.temperature} K")
    print(f"  Molecules: {indices}")
    print(f"  efpnorm → {args.device0}  |  rmsnorm → {args.device1}")
    print(f"{'='*70}\n")

    efp_dirs, rms_dirs = [], []

    def run_efp():
        efp_dirs.extend(run_batch("efpnorm", CHECKPOINTS["efpnorm"], args.device0, indices,
                                   args.steps, args.timestep, args.temperature,
                                   skip_existing=args.skip_existing))

    def run_rms():
        rms_dirs.extend(run_batch("rmsnorm", CHECKPOINTS["rmsnorm"], args.device1, indices,
                                   args.steps, args.timestep, args.temperature,
                                   skip_existing=args.skip_existing))

    t_efp = threading.Thread(target=run_efp)
    t_rms = threading.Thread(target=run_rms)
    t_efp.start(); t_rms.start()
    t_efp.join(); t_rms.join()

    def load_summary(label, idx):
        tag = out_dir_name(label, idx, args.steps, args.timestep, args.temperature)
        p = _ROOT / "eval" / "md_runs" / tag / "summary.json"
        if p.exists():
            with open(p) as f:
                return json.load(f)
        return {}

    print(f"\n{'='*70}")
    print(f"  SUMMARY: NVE energy drift per atom (meV/atom)  [lower = more stable]")
    print(f"  Duration: {ps:.2f} ps  T={args.temperature} K  dt={args.timestep} fs  (aimnet2 L4C128)")
    print(f"{'='*70}")
    print(f"  {'mol':>6}  {'formula':>26}  {'efpnorm':>12}  {'rmsnorm':>12}  {'winner':>10}")
    print("  " + "-" * 75)

    all_results = {}
    efp_wins = rms_wins = 0
    for idx in indices:
        efp = load_summary("efpnorm", idx)
        rms = load_summary("rmsnorm", idx)
        all_results[idx] = {"efpnorm": efp, "rmsnorm": rms}

        efp_drift = efp.get("energy_drift_per_atom_meV")
        rms_drift = rms.get("energy_drift_per_atom_meV")
        formula = efp.get("formula") or rms.get("formula") or "?"
        efp_str = f"{efp_drift:.3f}" if efp_drift is not None else "FAILED"
        rms_str = f"{rms_drift:.3f}" if rms_drift is not None else "FAILED"

        if efp_drift is not None and rms_drift is not None:
            winner = "efpnorm" if efp_drift < rms_drift else "rmsnorm"
            if winner == "efpnorm":
                efp_wins += 1
            else:
                rms_wins += 1
        else:
            winner = "-"
        print(f"  {idx:6d}  {formula:>26}  {efp_str:>12}  {rms_str:>12}  {winner:>10}")

    total = efp_wins + rms_wins
    print(f"\n  efpnorm wins: {efp_wins}/{total}  |  rmsnorm wins: {rms_wins}/{total}")
    print(f"{'='*70}\n")

    out_json = Path(args.out_json) if args.out_json else _ROOT / "eval" / "md_runs" / f"comparison_aimnet2_longpilot_{int(ps)}ps.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Combined results saved: {out_json}")


if __name__ == "__main__":
    main()
