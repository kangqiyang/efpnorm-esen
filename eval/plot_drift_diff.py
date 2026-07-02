"""
Paired per-molecule energy-drift difference vs time: drift_rmsnorm(t) -
drift_efpnorm(t), for the same 50-molecule qdpi strained set used in
plot_drift_curves.py.

Rationale: the plain overlay plot (plot_drift_curves.py) averages efpnorm
and rmsnorm separately across molecules, so a molecule where rmsnorm is much
worse and one where efpnorm is much worse cancel out in the population mean
even though something real happened in each. This plots the per-molecule
paired difference instead, so a nonzero-but-canceling effect would still be
visible as consistently-signed (or growing-over-time) individual lines, even
when the population mean sits near zero.

Positive value = rmsnorm drifted more (efpnorm more stable) at that instant.
Negative value = efpnorm drifted more (rmsnorm more stable).

Pairing is truncated to the shorter of the two runs' completed steps if one
side blew up early (failed run), since the difference is undefined once one
side has no more recorded steps.

Usage (from efpnorm-esen root):
    python eval/plot_drift_diff.py
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_ROOT = Path(__file__).parent.parent
_MD_RUNS = _ROOT / "eval" / "md_runs"
_OUT = _ROOT / "eval" / "figures" / "drift_diff_L4_vs_L6.png"

DEPTHS = ["L4C128", "L6C128"]

_STRAINED_INDICES = sorted(
    json.load(open(_MD_RUNS / "comparison_strained.json")).keys(), key=int
)


def load_pair_diff(depth, idx):
    efp_dir = _MD_RUNS / f"qdpi_{depth}_efpnorm_lr4e-4_val_s{idx}_T300_dt0.5_N5000_seed42"
    rms_dir = _MD_RUNS / f"qdpi_{depth}_rmsnorm_lr4e-4_val_s{idx}_T300_dt0.5_N5000_seed42"
    efp_csv, rms_csv = efp_dir / "md_metrics.csv", rms_dir / "md_metrics.csv"
    if not efp_csv.exists() or not rms_csv.exists():
        return None
    efp = pd.read_csv(efp_csv).set_index("step")["drift_per_atom"] * 1000.0
    rms = pd.read_csv(rms_csv).set_index("step")["drift_per_atom"] * 1000.0
    common = efp.index.intersection(rms.index)
    if len(common) == 0:
        return None
    diff = (rms.loc[common] - efp.loc[common])
    diff.index = common * 0.5 / 1000.0  # step -> ps
    return diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(_OUT))
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, depth in zip(axes, DEPTHS):
        diffs = []
        for idx in _STRAINED_INDICES:
            d = load_pair_diff(depth, idx)
            if d is not None:
                diffs.append(d)
                color = "tab:green" if d.iloc[-1] > 0 else "tab:red"
                ax.plot(d.index, d.values, color=color, alpha=0.25, linewidth=0.9)

        combined = pd.concat([d for d in diffs], axis=1)
        mean = combined.mean(axis=1)
        ax.plot(mean.index, mean.values, color="black", linewidth=2.5,
                 label=f"mean (n={len(diffs)})")
        ax.axhline(0, color="gray", linewidth=1, linestyle="--")
        ax.set_title(f"{depth}  (green: rmsnorm worse, red: efpnorm worse)")
        ax.set_xlabel("time (ps)")
        ax.legend()
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("drift_rmsnorm(t) - drift_efpnorm(t)  (meV/atom)")
    fig.suptitle("qdpi strained set: paired per-molecule drift difference vs time")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
