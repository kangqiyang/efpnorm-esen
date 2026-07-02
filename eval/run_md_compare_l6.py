"""
Run NVE MD on the L6C128 qdpi checkpoints (efpnorm vs rmsnorm) for the same
molecule set used in the original L4C128 comparison (eval/md_runs/comparison_summary.json:
val DB indices 0, 1, 2, 6, 10, 12), then print a side-by-side stability table.

Standalone counterpart to run_md_compare.py (which is pinned to L4C128) so that
script and its existing outputs are left untouched.

Usage (from efpnorm-esen root):
    python eval/run_md_compare_l6.py --device0 cuda:0 --device1 cuda:3
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
    "efpnorm": str(_ROOT / "train/checkpoints/qdpi_L6C128_efpnorm_lr4e-4"),
    "rmsnorm": str(_ROOT / "train/checkpoints/qdpi_L6C128_rmsnorm_lr4e-4"),
}


def out_dir_name(label, idx, steps, timestep, temperature):
    run_name = f"qdpi_L6C128_{label}_lr4e-4"
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
            "--dataset", "qdpi",
            "--split", "val",
            "--sample_idx", str(idx),
            "--temperature", str(temperature),
            "--timestep_fs", str(timestep),
            "--steps", str(steps),
            "--record_interval", "100",
            "--device", device,
            "--seed", "42",
            "--out_dir", str(out_dir),
        ]
        print(f"  [{label}] mol={idx} starting on {device}...", flush=True)
        t0 = time.time()
        result = subprocess.run(
            cmd, cwd=str(_ROOT),
            capture_output=True, text=True
        )
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
                    print(f"  [{label}] mol={idx} FAILED ({s.get('fail_reason','?')}) "
                          f"after {elapsed:.0f}s", flush=True)
                else:
                    print(f"  [{label}] mol={idx} drift={drift:.3f} meV/atom "
                          f"elapsed={elapsed:.0f}s", flush=True)
    return out_dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device0", default="cuda:0", help="GPU for efpnorm")
    parser.add_argument("--device1", default="cuda:1", help="GPU for rmsnorm")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--timestep", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--sample_indices", type=int, nargs="+",
                        default=[511, 23294, 521, 4823, 6065, 20725, 21769, 503, 2309, 397,
                                 13755, 2511, 16897, 7606, 21021, 17304, 23848, 19257, 5911,
                                 1914, 14007, 13167, 822, 17943, 20342, 1703, 7948, 20929,
                                 24695, 7271, 10966, 378, 7262, 16091, 13625, 6400, 19583,
                                 6303, 2545, 16456, 18554, 18655, 25646, 10120, 17052, 25486,
                                 11795, 15877, 8061, 7407],
                        help="Same 50-molecule 'strained' set used in the L4C128 comparison_strained.json run")
    parser.add_argument("--out_json", default=None,
                        help="Path for combined comparison JSON "
                             "(default: eval/md_runs/comparison_qdpi_L6_strained.json)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip molecules that already have a summary.json")
    args = parser.parse_args()

    indices = args.sample_indices
    ps = args.steps * args.timestep / 1000

    print(f"\n{'='*70}")
    print(f"  MD stability comparison: efpnorm vs rmsnorm  (dataset=qdpi, depth=L6)")
    print(f"  Steps={args.steps} × dt={args.timestep} fs = {ps:.2f} ps  T={args.temperature} K")
    print(f"  Molecules (val DB idx): {indices}")
    print(f"  efpnorm → {args.device0}  |  rmsnorm → {args.device1}")
    print(f"{'='*70}\n")

    efp_dirs = []
    rms_dirs = []

    def run_efp():
        efp_dirs.extend(run_batch("efpnorm", CHECKPOINTS["efpnorm"],
                                  args.device0, indices,
                                  args.steps, args.timestep, args.temperature,
                                  skip_existing=args.skip_existing))

    def run_rms():
        rms_dirs.extend(run_batch("rmsnorm", CHECKPOINTS["rmsnorm"],
                                  args.device1, indices,
                                  args.steps, args.timestep, args.temperature,
                                  skip_existing=args.skip_existing))

    t_efp = threading.Thread(target=run_efp)
    t_rms = threading.Thread(target=run_rms)
    t_efp.start()
    t_rms.start()
    t_efp.join()
    t_rms.join()

    def load_summary(label, idx):
        tag = out_dir_name(label, idx, args.steps, args.timestep, args.temperature)
        p = _ROOT / "eval" / "md_runs" / tag / "summary.json"
        if p.exists():
            with open(p) as f:
                return json.load(f)
        return {}

    print(f"\n{'='*70}")
    print(f"  SUMMARY: NVE energy drift per atom (meV/atom)  [lower = more stable]")
    print(f"  Duration: {ps:.2f} ps  T={args.temperature} K  dt={args.timestep} fs  (L6C128)")
    print(f"{'='*70}")
    print(f"  {'mol':>4}  {'formula':>20}  {'efpnorm':>12}  {'rmsnorm':>12}  {'winner':>10}")
    print("  " + "-" * 65)

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
            if efp_drift < rms_drift:
                winner = "efpnorm"
                efp_wins += 1
            else:
                winner = "rmsnorm"
                rms_wins += 1
        else:
            winner = "-"
        print(f"  {idx:4d}  {formula:>20}  {efp_str:>12}  {rms_str:>12}  {winner:>10}")

    total = efp_wins + rms_wins
    print(f"\n  efpnorm wins: {efp_wins}/{total}  |  rmsnorm wins: {rms_wins}/{total}")
    print(f"{'='*70}\n")

    out_json = Path(args.out_json) if args.out_json else _ROOT / "eval" / "md_runs" / "comparison_qdpi_L6_strained.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Combined results saved: {out_json}")


if __name__ == "__main__":
    main()
