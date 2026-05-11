"""
Convert the five processed HDF5 files to the unified schema expected by
PolyGET's HDF5 dataset reader:

  group/
    types   (n_atoms,)              int32   atomic numbers
    pos     (n_frames, n_atoms, 3)  float32 positions, Å
    energy  (n_frames,)             float64 atomization energy, eV
    forces  (n_frames, n_atoms, 3)  float32 forces, eV/Å

Input (processed/ with per-dataset field names):
  aimnet2_processed.h5  → numbers, coord, energy_atomization, forces
  ani2x_processed.h5    → species (n_frames,n_atoms), coordinates, energy_atomization, forces
  spice2_processed.h5   → atomic_numbers, conformations, formation_energy, forces
  spf_processed.h5      → numbers, pos, energy, forces  (closest to unified already)
  qdpi_processed.h5     → numbers, pos, energy_atomization, forces

Output: unified/ with one file per dataset, all in the schema above.

Energy weight is 0 in pretraining (force-only MSE), so the energy field is
stored for reader compatibility but does not affect training.
"""

import h5py
import numpy as np
from pathlib import Path
import time

SRC = Path(__file__).parent / "processed"
DST = Path(__file__).parent / "unified"
DST.mkdir(exist_ok=True)

DATASETS = {
    "aimnet2": {
        "src": SRC / "aimnet2_processed.h5",
        "types_key":  "numbers",
        "pos_key":    "coord",
        "energy_key": "energy_atomization",
        "forces_key": "forces",
        "types_2d":   True,   # numbers is (n_frames, n_atoms); take row 0
    },
    "ani2x": {
        "src": SRC / "ani2x_processed.h5",
        "types_key":  "species",
        "pos_key":    "coordinates",
        "energy_key": "energy_atomization",
        "forces_key": "forces",
        "types_2d":   True,   # species is (n_frames, n_atoms); take row 0
    },
    "spice2": {
        "src": SRC / "spice2_processed.h5",
        "types_key":  "atomic_numbers",
        "pos_key":    "conformations",
        "energy_key": "formation_energy",
        "forces_key": "forces",
        "types_2d":   False,
    },
    "spf": {
        "src": SRC / "spf_processed.h5",
        "types_key":  "numbers",
        "pos_key":    "pos",
        "energy_key": "energy",
        "forces_key": "forces",
        "types_2d":   False,
    },
    "qdpi": {
        "src": SRC / "qdpi_processed.h5",
        "types_key":  "numbers",
        "pos_key":    "pos",
        "energy_key": "energy_atomization",
        "forces_key": "forces",
        "types_2d":   False,
    },
}


def convert_dataset(name, cfg):
    src_path = cfg["src"]
    dst_path = DST / f"{name}.h5"

    if not src_path.exists():
        print(f"[{name}] SKIP — source not found: {src_path}")
        return

    if dst_path.exists():
        print(f"[{name}] SKIP — already exists: {dst_path}")
        return

    print(f"\n[{name}] Converting {src_path.name} → {dst_path.name}")
    t0 = time.time()
    n_groups = n_frames = 0

    with h5py.File(src_path, "r", locking=False) as src, \
         h5py.File(dst_path, "w", locking=False) as dst:

        def visit_group(name_in_file, obj):
            nonlocal n_groups, n_frames
            if not isinstance(obj, h5py.Group):
                return

            g = obj
            required = [cfg["types_key"], cfg["pos_key"], cfg["energy_key"], cfg["forces_key"]]
            if any(k not in g for k in required):
                return  # skip groups that don't have all fields (e.g. nested dirs)

            raw_types = g[cfg["types_key"]][()]
            if cfg["types_2d"]:
                # species is (n_frames, n_atoms) — all rows identical, take first
                types = raw_types[0].astype(np.int32)
            else:
                types = raw_types.astype(np.int32)

            pos    = g[cfg["pos_key"]][()].astype(np.float32)
            energy = g[cfg["energy_key"]][()].astype(np.float64)
            forces = g[cfg["forces_key"]][()].astype(np.float32)

            # Use a flat group name (replace '/' with '__' for nested QDpi groups)
            flat_name = name_in_file.replace("/", "__")
            dg = dst.require_group(flat_name)
            dg.create_dataset("types",  data=types,  compression="gzip", compression_opts=4)
            dg.create_dataset("pos",    data=pos,    compression="gzip", compression_opts=4)
            dg.create_dataset("energy", data=energy, compression="gzip", compression_opts=4)
            dg.create_dataset("forces", data=forces, compression="gzip", compression_opts=4)

            n_groups += 1
            n_frames += len(energy)

        src.visititems(visit_group)

    elapsed = time.time() - t0
    print(f"[{name}] Done — {n_groups:,} groups, {n_frames:,} frames in {elapsed:.0f}s")


def main():
    for name, cfg in DATASETS.items():
        convert_dataset(name, cfg)
    print("\nAll done. Unified files in:", DST)


if __name__ == "__main__":
    main()
