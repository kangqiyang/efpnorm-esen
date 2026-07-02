"""
Visualize the 20 ps AIMNet2 long-duration pilot (eval/run_md_longpilot.py):
8 stratified molecules (3 RMSNorm-worse outliers, 3 EFPNorm-worse outliers,
2 near-zero-gap "typical" controls at 2.5 ps), run out to 20 ps to test
whether the norm-scheme drift gap grows with trajectory duration or is
mostly noise.

Two figures:
  1. longpilot_drift_vs_time.png -- per-molecule drift-vs-time curves over
     the full 20 ps, small multiples, one panel per molecule.
  2. longpilot_gap_change.png -- paired scatter of the 2.5 ps gap vs the
     20 ps gap (rmsnorm - efpnorm drift) per molecule, with a y=x reference
     line, to directly show whether duration amplified or dampened each
     molecule's gap.

Usage (from efpnorm-esen root):
    python eval/plot_longpilot_results.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_ROOT = Path(__file__).parent.parent
_MD_RUNS = _ROOT / "eval" / "md_runs"
_FIGURES = _ROOT / "eval" / "figures"

# category label for each molecule, from the pilot's stratified selection
CATEGORY = {
    39411: "rms-worse", 253132: "rms-worse", 223835: "rms-worse",
    357430: "efp-worse", 244658: "efp-worse", 146045: "efp-worse",
    414475: "typical", 212470: "typical",
}

# 2.5 ps drift values, from the original 200-molecule AIMNet2 screen
# (eval/md_runs/comparison_aimnet2_full200.json), used to compute the "before" gap
SHORT_DRIFT_25PS = {
    39411:  (40.5, 280.1), 253132: (2.2, 31.2), 223835: (5.9, 26.2),
    357430: (31.9, 1.0), 244658: (23.0, 1.3), 146045: (47.8, 27.3),
    414475: (0.4, 0.6), 212470: (2.0, 2.4),
}  # sample_idx -> (efpnorm, rmsnorm) meV/atom


def run_dir(label, idx):
    return _MD_RUNS / f"aimnet2_L4C128_{label}_lr4e-4_val_s{idx}_T300_dt0.5_N40000_seed42"


def plot_drift_vs_time():
    indices = sorted(CATEGORY.keys())
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=False)
    for ax, idx in zip(axes.flat, indices):
        for label, color in [("efpnorm", "tab:blue"), ("rmsnorm", "tab:orange")]:
            csv_path = run_dir(label, idx) / "md_metrics.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            time_ps = df["step"] * 0.5 / 1000.0
            drift_meV = df["drift_per_atom"] * 1000.0
            ax.plot(time_ps, drift_meV, color=color, label=label, linewidth=1.3)
        ax.set_title(f"{idx}  ({CATEGORY[idx]})", fontsize=10)
        ax.set_xlabel("time (ps)")
        ax.set_ylabel("drift (meV/atom)")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("AIMNet2 20 ps long-duration pilot: energy drift vs time (8 stratified molecules)")
    fig.tight_layout()
    out = _FIGURES / "longpilot_drift_vs_time.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")


def plot_gap_change():
    pilot = json.load(open(_MD_RUNS / "comparison_aimnet2_longpilot_20ps.json"))
    fig, ax = plt.subplots(figsize=(7, 7))
    lims = [-50, 260]
    ax.plot(lims, lims, color="gray", linestyle="--", linewidth=1, label="gap unchanged (y=x)")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)

    for idx_str, entry in pilot.items():
        idx = int(idx_str)
        efp20 = entry["efpnorm"].get("energy_drift_per_atom_meV")
        rms20 = entry["rmsnorm"].get("energy_drift_per_atom_meV")
        if efp20 is None or rms20 is None:
            continue
        gap20 = rms20 - efp20
        efp25, rms25 = SHORT_DRIFT_25PS[idx]
        gap25 = rms25 - efp25
        color = {"rms-worse": "tab:green", "efp-worse": "tab:red", "typical": "tab:gray"}[CATEGORY[idx]]
        ax.scatter(gap25, gap20, color=color, s=60, zorder=3)
        ax.annotate(str(idx), (gap25, gap20), fontsize=8, xytext=(5, 5), textcoords="offset points")

    for cat, color in [("rms-worse", "tab:green"), ("efp-worse", "tab:red"), ("typical", "tab:gray")]:
        ax.scatter([], [], color=color, label=cat)  # legend entries only
    ax.set_xlabel("2.5 ps gap  (rmsnorm - efpnorm drift, meV/atom)")
    ax.set_ylabel("20 ps gap  (rmsnorm - efpnorm drift, meV/atom)")
    ax.set_title("Did the norm-scheme gap grow with trajectory duration?\n"
                  "(above y=x: gap grew favoring efpnorm; below: shrank or flipped)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = _FIGURES / "longpilot_gap_change.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")


if __name__ == "__main__":
    plot_drift_vs_time()
    plot_gap_change()
