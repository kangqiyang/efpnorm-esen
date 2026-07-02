"""
Run NVE MD on both rmsnorm and efpnorm checkpoints for a set of molecules,
then print a side-by-side stability comparison table.

Launches two subprocess batches in parallel (one per model / GPU) to keep
CUDA contexts isolated, then collects results from the per-run summary.json files.

Usage (from efpnorm-esen root):
    python eval/run_md_compare.py --device0 cuda:0 --device1 cuda:1 \\
        --steps 5000 --sample_indices 0 1 2 6 10 12
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent
_PYTHON = "/nethome/kyang394/scratch/envs/MLFF/bin/python"
_RUN_MD = str(_ROOT / "eval" / "run_md.py")

CHECKPOINTS = {
    "qdpi": {
        "efpnorm": str(_ROOT / "train/checkpoints/qdpi_L4C128_efpnorm_lr4e-4"),
        "rmsnorm": str(_ROOT / "train/checkpoints/qdpi_L4C128_rmsnorm_lr4e-4"),
    },
    "aimnet2": {
        "efpnorm": str(_ROOT / "train/checkpoints/aimnet2_L4C128_efpnorm_lr4e-4"),
        "rmsnorm": str(_ROOT / "train/checkpoints/aimnet2_L4C128_rmsnorm_lr4e-4"),
    },
}


def out_dir_name(dataset, label, idx, steps, timestep, temperature):
    run_name = f"{dataset}_L4C128_{label}_lr4e-4"
    return (f"{run_name}_val_s{idx}"
            f"_T{int(temperature)}_dt{timestep}_N{steps}_seed42")


def run_batch(label, ckpt, device, indices, steps, timestep, temperature, dataset, skip_existing=False):
    """Run run_md.py sequentially for each molecule index. Returns list of out dirs."""
    out_dirs = []
    for idx in indices:
        tag = out_dir_name(dataset, label, idx, steps, timestep, temperature)
        out_dir = _ROOT / "eval" / "md_runs" / tag
        out_dirs.append(out_dir)

        if skip_existing and (out_dir / "summary.json").exists():
            print(f"  [{label}] mol={idx} skipping (already done)", flush=True)
            continue

        cmd = [
            _PYTHON, _RUN_MD,
            "--checkpoint_dir", ckpt,
            "--dataset", dataset,
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
    parser.add_argument("--dataset", default="qdpi", choices=list(CHECKPOINTS),
                        help="Which pretrained model pair to compare")
    parser.add_argument("--device0", default="cuda:0", help="GPU for efpnorm")
    parser.add_argument("--device1", default="cuda:1", help="GPU for rmsnorm")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--timestep", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--sample_indices", type=int, nargs="+",
                        default=[0, 1, 2, 6, 10, 12])
    parser.add_argument("--out_json", default=None,
                        help="Path for combined comparison JSON (default: md_runs/comparison_<dataset>.json)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip molecules that already have a summary.json")
    args = parser.parse_args()

    indices = args.sample_indices
    ps = args.steps * args.timestep / 1000
    ckpts = CHECKPOINTS[args.dataset]

    print(f"\n{'='*70}")
    print(f"  MD stability comparison: efpnorm vs rmsnorm  (dataset={args.dataset})")
    print(f"  Steps={args.steps} × dt={args.timestep} fs = {ps:.2f} ps  T={args.temperature} K")
    print(f"  Molecules (val DB idx): {indices}")
    print(f"  efpnorm → {args.device0}  |  rmsnorm → {args.device1}")
    print(f"{'='*70}\n")

    # Launch both batches as threads so they run in parallel across GPUs
    import threading

    efp_dirs = []
    rms_dirs = []

    def run_efp():
        efp_dirs.extend(run_batch("efpnorm", ckpts["efpnorm"],
                                  args.device0, indices,
                                  args.steps, args.timestep, args.temperature,
                                  dataset=args.dataset,
                                  skip_existing=args.skip_existing))

    def run_rms():
        rms_dirs.extend(run_batch("rmsnorm", ckpts["rmsnorm"],
                                  args.device1, indices,
                                  args.steps, args.timestep, args.temperature,
                                  dataset=args.dataset,
                                  skip_existing=args.skip_existing))

    t_efp = threading.Thread(target=run_efp)
    t_rms = threading.Thread(target=run_rms)
    t_efp.start()
    t_rms.start()
    t_efp.join()
    t_rms.join()

    # Load all summaries
    def load_summary(label, idx):
        tag = out_dir_name(args.dataset, label, idx, args.steps, args.timestep, args.temperature)
        p = _ROOT / "eval" / "md_runs" / tag / "summary.json"
        if p.exists():
            with open(p) as f:
                return json.load(f)
        return {}

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"  SUMMARY: NVE energy drift per atom (meV/atom)  [lower = more stable]")
    print(f"  Duration: {ps:.2f} ps  T={args.temperature} K  dt={args.timestep} fs")
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

    # Save combined summary
    combined_path = Path(args.out_json) if args.out_json else _ROOT / "eval" / "md_runs" / f"comparison_{args.dataset}.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Combined results saved: {combined_path}")


if __name__ == "__main__":
    main()
