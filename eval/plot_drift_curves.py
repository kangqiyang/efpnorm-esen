"""
Plot energy-drift-vs-time curves (not just endpoint drift) for the qdpi
L4-vs-L6, efpnorm-vs-rmsnorm strained-set MD runs.

Reads md_metrics.csv from each of the 50x2x2 run directories under
eval/md_runs/qdpi_L{4,6}C128_{efpnorm,rmsnorm}_lr4e-4_val_s*_N5000_seed42,
and plots per-molecule traces (thin, transparent) plus the across-molecule
mean (bold) for each of the 4 (depth, norm) combinations. Failed/blown-up
runs contribute to the mean only while they were alive (their csv is
truncated at the failure step), so the mean at late times reflects only
surviving molecules.

Usage (from efpnorm-esen root):
    python eval/plot_drift_curves.py
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_ROOT = Path(__file__).parent.parent
_MD_RUNS = _ROOT / "eval" / "md_runs"
_OUT = _ROOT / "eval" / "figures" / "drift_vs_time_L4_vs_L6.png"

DEPTHS = ["L4C128", "L6C128"]
NORMS = ["efpnorm", "rmsnorm"]
COLORS = {"efpnorm": "tab:blue", "rmsnorm": "tab:orange"}

# Exact 50-molecule strained set, pinned from comparison_strained.json /
# comparison_qdpi_L6_strained.json (both use the same indices) so L4 and L6
# are compared on identical molecules, not just "whatever N5000 runs exist".
_STRAINED_INDICES = sorted(
    json.load(open(_MD_RUNS / "comparison_strained.json")).keys(), key=int
)


def load_traces(depth, norm):
    traces = []
    for idx in _STRAINED_INDICES:
        run_dir = _MD_RUNS / f"qdpi_{depth}_{norm}_lr4e-4_val_s{idx}_T300_dt0.5_N5000_seed42"
        csv_path = run_dir / "md_metrics.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        df["time_ps"] = df["step"] * 0.5 / 1000.0
        df["drift_meV"] = df["drift_per_atom"] * 1000.0
        traces.append(df[["time_ps", "drift_meV"]])
    return traces


def mean_curve(traces):
    combined = pd.concat(
        [t.set_index("time_ps")["drift_meV"] for t in traces], axis=1
    )
    return combined.mean(axis=1), combined.count(axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(_OUT))
    parser.add_argument("--spaghetti", action="store_true", default=True)
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, depth in zip(axes, DEPTHS):
        for norm in NORMS:
            traces = load_traces(depth, norm)
            if not traces:
                continue
            color = COLORS[norm]
            for t in traces:
                ax.plot(t["time_ps"], t["drift_meV"], color=color, alpha=0.12, linewidth=0.8)
            mean, n_alive = mean_curve(traces)
            ax.plot(mean.index, mean.values, color=color, linewidth=2.5,
                     label=f"{norm} (mean, n={len(traces)})")
        ax.set_title(depth)
        ax.set_xlabel("time (ps)")
        ax.set_ylim(0, 60)
        ax.legend()
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("energy drift |E(t)-E(0)| (meV/atom)")
    fig.suptitle("qdpi strained set (50 molecules): energy drift vs time, L4 vs L6")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
