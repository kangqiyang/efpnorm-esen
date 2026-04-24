"""
Preprocess QDpi wB97M-D3(BJ)/def2-TZVPPD dataset for MLFF pretraining.

Uses only the neutral/ split (6 files). The charged/ directory is skipped
entirely — charge state is encoded by file, not by a per-frame field.

Source format (DeePMD):
  Each file is an HDF5 with molecule-keyed groups. Per group:
    type_map.raw  (n_elem_types,) bytes  — element symbol strings
    type.raw      (n_atoms,)      int64  — index into type_map
    set.000/
      coord.npy   (N, n_atoms*3)  float64  Å (flattened)
      force.npy   (N, n_atoms*3)  float64  eV/Å (flattened)
      energy.npy  (N, 1)          float64  eV (total DFT)

Pipeline:
  1. Element filter  — keep molecules where ALL atoms are in ALLOWED_Z.
                       Most elements already within {H,C,N,O,F,S,Cl}; drops
                       any molecule with unresolvable type_map entry.

  2. Charge filter   — trivially passes: neutral/ files are all neutral.

  3. Energy filter   — drop conformers with total energy/atom outside
                       [E_MIN, E_MAX] Ha/atom. Energy is converted from eV
                       to Ha first for a consistent threshold with other datasets.

  4. Force filter    — drop conformers with any |F| > MAX_FORCE eV/Å.
                       Forces already in eV/Å (no conversion needed).

  5. Atomization     — subtract per-element reference energies (approximated
                       from AIMNet2 def2-TZVPP values; basis shift < 0.01 Ha/atom).

Output: processed/qdpi_processed.h5
  Groups keyed as "{filename}/{molecule_name}" (e.g. "ani/C0Cl0F0H10N0O5S0").
  The top-level groups are the file names (e.g. "ani"); each contains one
  subgroup per molecule.
  Datasets per group:
    numbers           (n_atoms,)              int32   atomic numbers
    pos               (n_frames, n_atoms, 3)  float32 positions in Å
    forces            (n_frames, n_atoms, 3)  float32 forces in eV/Å
    energy            (n_frames,)             float64 total DFT energy in eV
    energy_atomization (n_frames,)            float64 atomization energy in eV
"""

import h5py
import numpy as np
from pathlib import Path
import time

# ── Paths ────────────────────────────────────────────────────────────────
QDPI_BASE = Path("raw/qdpi/QDpiDataset-main/data/neutral")
QDPI_FILES = ["ani", "comp6", "freesolvmd", "geom", "re", "remd"]
OUTPUT     = Path("processed/qdpi_processed.h5")

# ── Filter parameters ────────────────────────────────────────────────────
ALLOWED_Z  = frozenset([1, 6, 7, 8, 9, 14, 15, 16, 17, 35])
MAX_FORCE  = 20.0     # eV/Å
E_MIN      = -1000.0  # Ha/atom  (applied to total energy)
E_MAX      = -0.4     # Ha/atom

# ── Unit conversions ─────────────────────────────────────────────────────
HA_TO_EV = 27.211386

# ── Element type-map used by QDpi ────────────────────────────────────────
QDPI_SYMBOL_TO_Z = {
    "H": 1, "C": 6, "N": 7, "O": 8, "F": 9,
    "S": 16, "Cl": 17, "P": 15, "Si": 14, "Br": 35,
}

# ── Atomization reference energies (eV) — wB97M-D3(BJ)/def2-TZVPP proxy ─
# Approximated from AIMNet2 def2-TZVPP values; basis shift < 0.01 Ha/atom.
# !!VERIFY if using a different theory level!!
_ANI2X_REF_HA = {
    1:  -0.500607, 6: -37.845355, 7: -54.582445, 8:  -75.062826,
    9: -99.716370, 16: -398.088185, 17: -460.135649,
}
REFERENCE_ENERGIES_EV = {z: e * HA_TO_EV for z, e in _ANI2X_REF_HA.items()}


def decode_type_map(type_map_raw: np.ndarray) -> dict[int, int]:
    """Map type-index → atomic number using the per-molecule type_map."""
    symbols = [
        x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)
        for x in type_map_raw
    ]
    return {i: QDPI_SYMBOL_TO_Z.get(sym, 0) for i, sym in enumerate(symbols)}


def process():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_in":      0,
        "dropped_elem":  0,
        "dropped_energy": 0,
        "dropped_force": 0,
        "total_out":     0,
    }

    with h5py.File(OUTPUT, "w", locking=False) as fout:
        fout.attrs["source"]         = str(QDPI_BASE)
        fout.attrs["files"]          = QDPI_FILES
        fout.attrs["allowed_Z"]      = sorted(ALLOWED_Z)
        fout.attrs["max_force_evA"]  = MAX_FORCE
        fout.attrs["e_min_ha_atom"]  = E_MIN
        fout.attrs["e_max_ha_atom"]  = E_MAX
        fout.attrs["energy_unit"]    = "eV (total DFT); energy_atomization in eV"
        fout.attrs["force_unit"]     = "eV/Ang"
        fout.attrs["coord_unit"]     = "Ang"

        kw = dict(compression="gzip", compression_opts=4)

        for fname in QDPI_FILES:
            fpath = QDPI_BASE / (fname + ".hdf5")
            print(f"\n  {fname}.hdf5")
            with h5py.File(fpath, "r", locking=False) as fin:
                mols = sorted(fin.keys())
                for i_mol, mol in enumerate(mols):
                    g        = fin[mol]
                    idx_to_z = decode_type_map(g["type_map.raw"][:])
                    zs       = np.array([idx_to_z.get(int(t), 0)
                                         for t in g["type.raw"][:]], dtype=np.int32)
                    n_atoms  = len(zs)

                    # collect all conformers across set.NNN subgroups
                    sets = sorted(s for s in g.keys() if s.startswith("set."))
                    energy_list, coord_list, force_list = [], [], []
                    for s in sets:
                        energy_list.append(g[s + "/energy.npy"][:].flatten())
                        coord_list.append(
                            g[s + "/coord.npy"][:].reshape(-1, n_atoms, 3)
                        )
                        force_list.append(
                            g[s + "/force.npy"][:].reshape(-1, n_atoms, 3)
                        )

                    energy = np.concatenate(energy_list)   # (N,) eV  total DFT
                    coords = np.concatenate(coord_list)    # (N, n_atoms, 3) Å
                    forces = np.concatenate(force_list)    # (N, n_atoms, 3) eV/Å
                    N      = len(energy)
                    stats["total_in"] += N

                    # 1. element filter (molecule-level)
                    if not np.isin(zs, sorted(ALLOWED_Z)).all():
                        stats["dropped_elem"] += N
                        continue

                    # 3. energy filter (convert eV → Ha for consistent threshold)
                    epa_ha    = (energy / HA_TO_EV) / n_atoms
                    energy_ok = (epa_ha >= E_MIN) & (epa_ha <= E_MAX)

                    # 4. force filter
                    fmag_max = np.linalg.norm(forces, axis=-1).max(axis=1)
                    force_ok = fmag_max <= MAX_FORCE

                    keep   = energy_ok & force_ok
                    n_keep = int(keep.sum())

                    stats["dropped_energy"] += int((~energy_ok).sum())
                    stats["dropped_force"]  += int((energy_ok & ~force_ok).sum())
                    stats["total_out"]      += n_keep

                    if (i_mol + 1) % 5000 == 0:
                        print(f"    [{i_mol+1:>5}/{len(mols):>5}]  {mol:<35}"
                              f"  {N:>5} in  => {n_keep:>5} kept")

                    if n_keep == 0:
                        continue

                    # 5. atomization energy
                    ref_sum     = sum(REFERENCE_ENERGIES_EV.get(int(z), 0.0) for z in zs)
                    e_atomiz    = energy[keep] - ref_sum  # eV

                    file_grp = fout.require_group(fname)
                    out = file_grp.create_group(mol)
                    out.create_dataset("numbers",            data=zs,                                      **kw)
                    out.create_dataset("pos",                data=coords[keep].astype(np.float32),         **kw)
                    out.create_dataset("forces",             data=forces[keep].astype(np.float32),         **kw)
                    out.create_dataset("energy",             data=energy[keep].astype(np.float64),         **kw)
                    out.create_dataset("energy_atomization", data=e_atomiz.astype(np.float64),             **kw)

    return stats


if __name__ == "__main__":
    t0 = time.time()
    print(f"Input : {QDPI_BASE}")
    print(f"Files : {QDPI_FILES}")
    print(f"Output: {OUTPUT}")
    print(f"Max force : {MAX_FORCE} eV/Å")
    print(f"E/atom    : [{E_MIN}, {E_MAX}] Ha\n")

    stats = process()

    total_in  = stats["total_in"]
    total_out = stats["total_out"]
    print()
    print(f"{'Input conformers':<32}: {total_in:>12,}")
    print(f"{'Dropped (non-allowed elements)':<32}: {stats['dropped_elem']:>12,}  ({100*stats['dropped_elem']/total_in:.1f}%)")
    print(f"{'Dropped (energy outlier)':<32}: {stats['dropped_energy']:>12,}  ({100*stats['dropped_energy']/total_in:.1f}%)")
    print(f"{'Dropped (force > 20 eV/Å)':<32}: {stats['dropped_force']:>12,}  ({100*stats['dropped_force']/total_in:.1f}%)")
    print(f"{'Output conformers':<32}: {total_out:>12,}  ({100*total_out/total_in:.1f}%)")
    print(f"\nDone in {time.time() - t0:.1f}s")
    print(f"Written to: {OUTPUT}")
