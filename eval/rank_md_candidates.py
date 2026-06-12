"""
Rank val-set structures by reference DFT force magnitude to identify
good NVE-MD benchmark candidates without running any model inference.

Low-force structures (max_force_norm < 1.5 eV/Å) are near equilibrium
and give steady-state drift numbers.  High-force structures are stress-test
geometries useful for blow-up / NaN robustness tests.

Usage
-----
    python eval/rank_md_candidates.py --dataset qdpi --split val --top_k 20
    python eval/rank_md_candidates.py --dataset aimnet2 --split val --top_k 50 \\
        --fmax_lo 1.0 --fmax_hi 3.0

Output
------
    Console: rich table of top_k structures sorted ascending by max_force_norm.
    CSV    : eval/candidate_ranks/<dataset>_<split>_candidates.csv  (full ranking)
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from ase.db import connect

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn
from rich.table import Table
from rich.rule import Rule

_ROOT    = Path(__file__).parent.parent
DATA_DIR = _ROOT / "data" / "asedb"
OUT_DIR  = _ROOT / "eval" / "candidate_ranks"

DATASETS = ["aimnet2", "spice2", "qdpi", "ani2x", "spf"]

console = Console(highlight=False)


def rank_candidates(db_path: Path, fmax_lo: float, fmax_hi: float) -> list[dict]:
    """Return one dict per row, sorted ascending by max_force_norm."""
    rows = []
    with connect(str(db_path)) as db:
        total = db.count()
        with Progress(
            SpinnerColumn(),
            BarColumn(bar_width=32),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console, transient=True,
        ) as prog:
            task = prog.add_task("", total=total)
            for sample_idx, row in enumerate(db.select()):
                atoms   = row.toatoms()
                forces  = atoms.get_forces()          # (N, 3) in eV/Å
                fnorms  = np.linalg.norm(forces, axis=1)  # (N,)
                max_f   = float(fnorms.max())
                rms_f   = float(np.sqrt(np.mean(fnorms ** 2)))
                flag    = ("equilibrated" if max_f < fmax_lo
                           else "strained" if max_f > fmax_hi
                           else "moderate")
                rows.append({
                    "sample_idx":    sample_idx,
                    "formula":       atoms.get_chemical_formula("hill"),
                    "n_atoms":       len(atoms),
                    "max_force_norm": round(max_f, 4),
                    "rms_force":      round(rms_f, 4),
                    "flag":          flag,
                })
                prog.advance(task)

    rows.sort(key=lambda r: r["max_force_norm"])
    return rows


def print_table(rows: list[dict], top_k: int, fmax_lo: float, fmax_hi: float) -> None:
    flag_style = {"equilibrated": "green", "moderate": "cyan", "strained": "red"}

    t = Table(show_header=True, header_style="bold", box=None, pad_edge=False, min_width=72)
    t.add_column("idx",          style="dim",    justify="right", width=6)
    t.add_column("formula",      justify="left",  width=14)
    t.add_column("n_atoms",      justify="right", width=8)
    t.add_column("max_F (eV/Å)", justify="right", width=14)
    t.add_column("rms_F (eV/Å)", justify="right", width=14)
    t.add_column("flag",         justify="left",  width=14)

    for r in rows[:top_k]:
        style = flag_style[r["flag"]]
        t.add_row(
            str(r["sample_idx"]),
            r["formula"],
            str(r["n_atoms"]),
            f"{r['max_force_norm']:.4f}",
            f"{r['rms_force']:.4f}",
            f"[{style}]{r['flag']}[/{style}]",
        )

    console.print(t)

    n_eq  = sum(1 for r in rows if r["flag"] == "equilibrated")
    n_st  = sum(1 for r in rows if r["flag"] == "strained")
    n_mod = len(rows) - n_eq - n_st
    console.print(
        f"\n[dim]Total {len(rows)} structures —[/] "
        f"[green]{n_eq} equilibrated[/] (max_F < {fmax_lo} eV/Å)  "
        f"[cyan]{n_mod} moderate[/]  "
        f"[red]{n_st} strained[/] (max_F > {fmax_hi} eV/Å)"
    )


def save_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample_idx", "formula", "n_atoms", "max_force_norm", "rms_force", "flag"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    console.print(f"\n[dim]saved →[/] {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, default="qdpi")
    parser.add_argument("--split",   choices=["train", "val"], default="val")
    parser.add_argument("--top_k",   type=int,   default=20)
    parser.add_argument("--fmax_lo", type=float, default=1.5,
                        help="max_force_norm threshold for 'equilibrated' (eV/Å)")
    parser.add_argument("--fmax_hi", type=float, default=5.0,
                        help="max_force_norm threshold for 'strained' (eV/Å)")
    args = parser.parse_args()

    db_path = DATA_DIR / f"{args.dataset}_{args.split}.db"
    if not db_path.exists():
        console.print(f"[red]DB not found:[/] {db_path}")
        sys.exit(1)

    console.print(Rule(f"[dim]{args.dataset} / {args.split}[/]", style="dim"))
    rows = rank_candidates(db_path, args.fmax_lo, args.fmax_hi)

    console.print(Rule(f"[dim]top {args.top_k} by max_force_norm[/]", style="dim"))
    print_table(rows, args.top_k, args.fmax_lo, args.fmax_hi)

    out_path = OUT_DIR / f"{args.dataset}_{args.split}_candidates.csv"
    save_csv(rows, out_path)


if __name__ == "__main__":
    main()
