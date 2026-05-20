"""
h5 (unified schema) → ASE SQLite database

Unified h5 schema (all datasets):
  group/
    types   (n_atoms,)              int32   atomic numbers
    pos     (n_frames, n_atoms, 3)  float32 positions, Å
    energy  (n_frames,)             float64 atomization/formation energy, eV
    forces  (n_frames, n_atoms, 3)  float32 forces, eV/Å

Output: one ASE .db file, each row is one frame (non-periodic Atoms +
SinglePointCalculator with energy and forces).

Usage:
    python h5_to_asedb.py --dataset aimnet2 --max_frames 10000 --val_frac 0.05

"""

import argparse
import random
import time
from pathlib import Path

import h5py
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect

DATA_DIR = Path(__file__).parent / "unified"
OUT_DIR  = Path(__file__).parent / "asedb"

DATASETS = ["aimnet2", "spice2", "qdpi", "ani2x", "spf"]


def iter_frames(h5_path: Path):
    """Yield (atomic_numbers, positions, energy, forces) for every frame.

    Handles two unified-H5 group layouts:
      types (n_atoms,)            — one fixed formula per group (spice2/spf/qdpi)
      types (n_frames, n_atoms)   — atom-count bucket groups (aimnet2/ani2x);
                                    split on-the-fly by unique molecular formula.
    """
    _required = {"types", "pos", "energy", "forces"}
    with h5py.File(h5_path, "r", locking=False) as f:
        groups = []
        f.visititems(lambda name, obj: groups.append(name)
                     if isinstance(obj, h5py.Group) and _required.issubset(obj.keys())
                     else None)
        for grp_name in groups:
            grp = f[grp_name]
            Z   = grp["types"][()]   # (n_atoms,) or (n_frames, n_atoms)
            pos = grp["pos"][()]     # (n_frames, n_atoms, 3)
            E   = grp["energy"][()]  # (n_frames,)
            F   = grp["forces"][()]  # (n_frames, n_atoms, 3)

            if Z.ndim == 1:
                # One formula for all frames in this group.
                for i in range(len(E)):
                    yield Z, pos[i], float(E[i]), F[i]
            elif Z.ndim == 2:
                # Atom-count bucket: split by unique molecular formula.
                unique_types, inverse = np.unique(
                    Z.astype(np.int32), axis=0, return_inverse=True
                )
                for mol_idx, formula in enumerate(unique_types):
                    mask = inverse == mol_idx
                    idxs = np.where(mask)[0]
                    for i in idxs:
                        yield formula, pos[i], float(E[i]), F[i]
            else:
                raise ValueError(
                    f"Group '{grp_name}' in {h5_path.name}: "
                    f"unexpected types shape {Z.shape}"
                )


def write_db(frames, db_path: Path, dataset_name: str):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with connect(str(db_path), append=False) as db:
        for Z, pos, E, F in frames:
            atoms = Atoms(numbers=Z, positions=pos, pbc=False)
            atoms.calc = SinglePointCalculator(atoms, energy=E, forces=F)
            db.write(atoms, data={"dataset": dataset_name})
    elapsed = time.time() - t0
    return elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS + ["all"], default="aimnet2",
                        help="Which dataset to convert (default: aimnet2)")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Cap total frames (for quick tests)")
    parser.add_argument("--val_frac", type=float, default=0.05,
                        help="Fraction of frames held out for validation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else [args.dataset]

    for name in datasets:
        h5_path = DATA_DIR / f"{name}.h5"
        if not h5_path.exists():
            print(f"[{name}] SKIP — {h5_path} not found")
            continue

        print(f"\n[{name}] Reading frames from {h5_path} ...")
        all_frames = list(iter_frames(h5_path))
        print(f"[{name}] Total frames: {len(all_frames):,}")

        if args.max_frames and len(all_frames) > args.max_frames:
            rng = random.Random(args.seed)
            all_frames = rng.sample(all_frames, args.max_frames)
            print(f"[{name}] Subsampled to {len(all_frames):,} frames")

        rng = random.Random(args.seed)
        rng.shuffle(all_frames)
        n_val   = max(1, int(len(all_frames) * args.val_frac))
        n_train = len(all_frames) - n_val
        train_frames = all_frames[:n_train]
        val_frames   = all_frames[n_train:]

        train_db = OUT_DIR / f"{name}_train.db"
        val_db   = OUT_DIR / f"{name}_val.db"

        print(f"[{name}] Writing train ({n_train:,} frames) → {train_db}")
        t = write_db(train_frames, train_db, name)
        print(f"[{name}] Train done in {t:.1f}s")

        print(f"[{name}] Writing val   ({n_val:,} frames) → {val_db}")
        t = write_db(val_frames, val_db, name)
        print(f"[{name}] Val   done in {t:.1f}s")

    print("\nDone. Files in:", OUT_DIR)


if __name__ == "__main__":
    main()
