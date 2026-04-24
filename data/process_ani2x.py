"""
Preprocess ANI-2x wB97X/6-31G(d) dataset for MLFF pretraining.

Pipeline:
  1. Element filter  — trivially passes: ANI-2x contains only {H, C, N, O, F, S, Cl},
                       all within ALLOWED_Z.

  2. Charge filter   — trivially passes: all ANI-2x molecules are neutral by construction.

  3. Energy filter   — drop conformers with total energy/atom outside [E_MIN, E_MAX] Ha/atom.

  4. Force filter    — drop conformers with any |F| > MAX_FORCE eV/Å.
                       Forces stored as true forces in Ha/Å → convert by × HA_TO_EV.

  5. Subsampling     — cap at MAX_PER_COMP conformers per unique molecular composition
                       (defined by exact species array). Prevents e.g. H₄O₂ (60k
                       conformers) from dominating. Reduces ~9.65M → ~542k frames.

  6. Atomization     — subtract per-element reference energies at wB97X/6-31G(d).
                       Source: torchani.models.ANI2x self_energies.

Output: processed/ani2x_processed.h5
  Groups keyed by atom count string (same as input, e.g. '007').
  Datasets: species (n_keep, n_atoms), coordinates (n_keep, n_atoms, 3),
            forces (n_keep, n_atoms, 3), energies (n_keep,), energy_atomization (n_keep,).
  forces in eV/Å; energies in Ha (original); energy_atomization in eV.
"""

import h5py
import numpy as np
from collections import defaultdict
from pathlib import Path
import time

# ── Paths ────────────────────────────────────────────────────────────────
INPUT  = Path("raw/ani2x/final_h5/ANI-2x-wB97X-631Gd.h5")
OUTPUT = Path("processed/ani2x_processed.h5")

# ── Filter parameters ────────────────────────────────────────────────────
MAX_FORCE      = 20.0     # eV/Å
E_MIN          = -1000.0  # Ha/atom
E_MAX          = -0.4     # Ha/atom
MAX_PER_COMP   = 100      # max conformers per unique composition

# ── Unit conversions ─────────────────────────────────────────────────────
HA_TO_EV = 27.211386

# ── Atomization reference energies (Ha) — wB97X/6-31G(d) ────────────────
# Source: torchani.models.ANI2x self_energies
REFERENCE_ENERGIES = {
    1:  -0.500607,   # H
    6:  -37.845355,  # C
    7:  -54.582445,  # N
    8:  -75.062826,  # O
    9:  -99.716370,  # F
    16: -398.088185, # S
    17: -460.135649, # Cl
}


def compute_atomization(energies_ha: np.ndarray, species: np.ndarray) -> np.ndarray:
    """
    energies_ha : (N_conf,)          total DFT energies in Hartree
    species     : (N_conf, n_atoms)  atomic numbers
    returns     : (N_conf,)          atomization energies in eV
    """
    ref = np.array([
        sum(REFERENCE_ENERGIES.get(int(z), 0.0) for z in species[i])
        for i in range(len(energies_ha))
    ])
    return (energies_ha - ref) * HA_TO_EV


def subsample_by_composition(species: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return indices that cap each unique species-tuple at MAX_PER_COMP."""
    comp_to_idx: dict[tuple, list[int]] = defaultdict(list)
    for i, row in enumerate(species):
        comp_to_idx[tuple(row.tolist())].append(i)

    keep = []
    for indices in comp_to_idx.values():
        if len(indices) > MAX_PER_COMP:
            chosen = rng.choice(indices, MAX_PER_COMP, replace=False)
            keep.extend(chosen.tolist())
        else:
            keep.extend(indices)
    return np.array(sorted(keep), dtype=np.int64)


def process():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    stats = {
        "total_in": 0,
        "dropped_energy": 0,
        "dropped_force": 0,
        "dropped_subsample": 0,
        "total_out": 0,
    }

    with h5py.File(INPUT, "r", locking=False) as fin, \
         h5py.File(OUTPUT, "w", locking=False) as fout:
        fout.attrs["source"]          = str(INPUT)
        fout.attrs["max_force_evA"]   = MAX_FORCE
        fout.attrs["e_min_ha_atom"]   = E_MIN
        fout.attrs["e_max_ha_atom"]   = E_MAX
        fout.attrs["max_per_comp"]    = MAX_PER_COMP
        fout.attrs["energy_unit"]     = "Ha (original); energy_atomization in eV"
        fout.attrs["force_unit"]      = "eV/Ang"
        fout.attrs["coord_unit"]      = "Ang"

        for g_key in sorted(fin.keys()):
            grp      = fin[g_key]
            n_atoms  = int(g_key)
            sp       = grp["species"][:]      # (N_conf, n_atoms) int64
            coords   = grp["coordinates"][:]  # (N_conf, n_atoms, 3) float32 Å
            energies = grp["energies"][:]     # (N_conf,) float64 Ha
            forces_ha = grp["forces"][:]      # (N_conf, n_atoms, 3) float64 Ha/Å
            N = sp.shape[0]
            stats["total_in"] += N

            # 3. energy filter
            epa         = energies / n_atoms
            energy_mask = (epa >= E_MIN) & (epa <= E_MAX)

            # 4. force filter (convert Ha/Å → eV/Å)
            forces_ev  = forces_ha * HA_TO_EV
            fmag_max   = np.linalg.norm(forces_ev, axis=-1).max(axis=1)
            force_mask = fmag_max <= MAX_FORCE

            quality_mask = energy_mask & force_mask
            n_quality    = int(quality_mask.sum())

            stats["dropped_energy"] += int((~energy_mask).sum())
            stats["dropped_force"]  += int((energy_mask & ~force_mask).sum())

            # 5. subsample by composition among quality-passing conformers
            q_idx    = np.where(quality_mask)[0]
            sp_q     = sp[q_idx]
            sub_idx  = subsample_by_composition(sp_q, rng)
            keep_idx = q_idx[sub_idx]
            n_keep   = len(keep_idx)

            stats["dropped_subsample"] += n_quality - n_keep
            stats["total_out"]         += n_keep

            print(
                f"  {g_key}: {N:>8,} in"
                f"  -ene {int((~energy_mask).sum()):>6,}"
                f"  -frc {int((energy_mask & ~force_mask).sum()):>6,}"
                f"  -sub {n_quality - n_keep:>6,}"
                f"  => {n_keep:>8,} kept"
            )

            if n_keep == 0:
                continue

            # 6. compute atomization energy for kept conformers
            e_atom = compute_atomization(energies[keep_idx], sp[keep_idx])

            out = fout.create_group(g_key)
            kw  = dict(compression="gzip", compression_opts=4)

            out.create_dataset("species",            data=sp[keep_idx].astype(np.int32),          **kw)
            out.create_dataset("coordinates",        data=coords[keep_idx].astype(np.float32),     **kw)
            out.create_dataset("forces",             data=forces_ev[keep_idx].astype(np.float32),  **kw)
            out.create_dataset("energies",           data=energies[keep_idx],                      **kw)
            out.create_dataset("energy_atomization", data=e_atom.astype(np.float64),               **kw)

    return stats


if __name__ == "__main__":
    t0 = time.time()
    print(f"Input : {INPUT}")
    print(f"Output: {OUTPUT}")
    print(f"Max force      : {MAX_FORCE} eV/Å")
    print(f"E/atom         : [{E_MIN}, {E_MAX}] Ha")
    print(f"Max per comp   : {MAX_PER_COMP}\n")

    stats = process()

    total_in  = stats["total_in"]
    total_out = stats["total_out"]
    print()
    print(f"{'Input conformers':<32}: {total_in:>12,}")
    print(f"{'Dropped (energy outlier)':<32}: {stats['dropped_energy']:>12,}  ({100*stats['dropped_energy']/total_in:.1f}%)")
    print(f"{'Dropped (force > 20 eV/Å)':<32}: {stats['dropped_force']:>12,}  ({100*stats['dropped_force']/total_in:.1f}%)")
    print(f"{'Dropped (subsampling cap)':<32}: {stats['dropped_subsample']:>12,}  ({100*stats['dropped_subsample']/total_in:.1f}%)")
    print(f"{'Output conformers':<32}: {total_out:>12,}  ({100*total_out/total_in:.1f}%)")
    print(f"\nDone in {time.time() - t0:.1f}s")
    print(f"Written to: {OUTPUT}")
