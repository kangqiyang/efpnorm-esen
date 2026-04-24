"""
Preprocess AIMNet2 wB97M-D3(BJ)/def2-TZVPP dataset for MLFF pretraining.

Pipeline:
  1. Element filter  — keep conformers where ALL atoms are in ALLOWED_Z
                       {H, C, N, O, F, Si, P, S, Cl, Br}

       Rationale: expanded beyond strict CHONS to include common polymer-relevant
       elements (F in fluoropolymers, Cl/Br in halogenated monomers, Si in
       silicones, P in flame retardants). Truly rare elements with no polymer
       relevance (B, As, Se, I) are dropped.

  2. Charge filter   — keep charge == 0 (neutral only)

       Rationale: the model takes only atomic numbers (z) and positions (pos) as
       inputs — there is no charge input feature. This has two consequences:

       (a) For radical ions (same z, electron added/removed): the model cannot
           distinguish a charged conformer from a neutral one with similar geometry.
           Including these conformers adds force/energy targets the model has no
           signal to predict correctly — wasted capacity and gradient noise.

       (b) For protonated/deprotonated species (different z due to extra/missing H):
           the geometry does encode the charge state, so there is no fundamental
           ambiguity. However, the fine-tuning target (CHONS VASP polymer AIMD) is
           entirely neutral. Including charged species would shift the training
           distribution toward ionic chemistry the model will never encounter
           downstream, spending capacity on irrelevant regimes.

       If charge is added as a model input in the future, this filter should be
       revisited — charged molecules can then be included without ambiguity.

  3. Energy filter   — keep energy/atom within [E_MIN, E_MAX] (Ha/atom)
                       removes corrupted DFT entries (e.g. S2 dimer at -10836 Ha/atom,
                       which is ~27x below the physical S atomic energy of -398 Ha)

  4. Force filter    — drop conformers with any |F| > MAX_FORCE (eV/Å)
                       0.024% of atom-frames exceed 20 eV/Å; these dominate MSE
                       loss (scales as |F|²) without representing polymer-relevant
                       chemistry at 300K

  5. Atomization     — subtract per-element reference energies (Ha) then convert to eV
                       raw DFT total energies (~10³ Ha) are not model-learnable;
                       atomization energies (~1–100 eV) are

  6. Unit conversion — energy Ha -> eV; forces already eV/Å; coords stay Å

Output: processed/aimnet2_processed.h5
  Same group structure as input (keyed by zero-padded atom count).
  New dataset 'energy_atomization' (eV) added alongside original 'energy' (Ha).
  Only groups with at least one passing conformer are written.

Reference energies: wB97M-D3(BJ)/def2-TZVPP spin-averaged atomic energies (Ha).
Source: AIMNet2 paper (Anstine et al., JCTC 2023) + standard atomic calculations.
!!VERIFY these before use if replicating at a different theory level!!
"""

import h5py
import numpy as np
from pathlib import Path
import time

# ── Paths ────────────────────────────────────────────────────────────────
INPUT  = Path("raw/aimnet2/aimnet2_wb97m.h5")
OUTPUT = Path("processed/aimnet2_processed.h5")

# ── Filter parameters ────────────────────────────────────────────────────
# Expanded beyond CHONS to keep common polymer-relevant elements
ALLOWED_Z  = frozenset([1, 6, 7, 8, 9, 14, 15, 16, 17, 35])
#                       H  C  N  O  F  Si   P   S  Cl  Br
MAX_CHARGE = 0   # neutral only — see module docstring for full rationale
MAX_FORCE  = 20.0   # eV/Å — p99.9 of dataset is 26.5, max is 96.7
E_MIN      = -1000.0  # Ha/atom  (CHONS range: H~-0.5, S~-398)
E_MAX      = -0.4     # Ha/atom  (just below H atomic energy)

# ── Atomization reference energies (Ha) — wB97M-D3(BJ)/def2-TZVPP ───────
REFERENCE_ENERGIES = {
    1:  -0.500607,   # H
    6:  -37.846772,  # C
    7:  -54.583861,  # N
    8:  -75.064579,  # O
    9:  -99.718730,  # F
    14: -289.359782, # Si
    15: -341.259942, # P
    16: -397.897380, # S
    17: -460.117861, # Cl
    35: -2573.966,   # Br  ← verify; Br is heavy, ECP may shift this
}

HA_TO_EV = 27.211386  # CODATA 2018

ELEM_SYMBOL = {
    1: "H", 6: "C", 7: "N", 8: "O", 9: "F",
    14: "Si", 15: "P", 16: "S", 17: "Cl", 35: "Br",
}


def compute_atomization(energy_ha: np.ndarray, numbers: np.ndarray) -> np.ndarray:
    """
    energy_ha : (N_conf,)          total DFT energies in Hartree
    numbers   : (N_conf, n_atoms)  atomic numbers
    returns   : (N_conf,)          atomization energies in eV
    """
    ref = np.array([
        sum(REFERENCE_ENERGIES[int(z)] for z in numbers[i])
        for i in range(len(energy_ha))
    ])
    return (energy_ha - ref) * HA_TO_EV


def build_elem_mask(numbers: np.ndarray) -> np.ndarray:
    """Vectorized: True where all atoms in conformer are in ALLOWED_Z."""
    # numbers: (N_conf, n_atoms)
    flat_allowed = np.array(sorted(ALLOWED_Z), dtype=np.int16)
    # for each atom check membership, then all() per conformer
    in_allowed = np.isin(numbers, flat_allowed)   # (N_conf, n_atoms)
    return in_allowed.all(axis=1)                  # (N_conf,)


def process():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_in": 0,
        "dropped_elem": 0,
        "dropped_charge": 0,
        "dropped_energy": 0,
        "dropped_force": 0,
        "total_out": 0,
    }

    with h5py.File(INPUT, "r", locking=False) as fin, \
         h5py.File(OUTPUT, "w", locking=False) as fout:

        # store filter params as file-level attrs for reproducibility
        fout.attrs["source"]         = str(INPUT)
        fout.attrs["allowed_Z"]      = sorted(ALLOWED_Z)
        fout.attrs["max_charge"]     = MAX_CHARGE
        fout.attrs["max_force_evA"]  = MAX_FORCE
        fout.attrs["e_min_ha_atom"]  = E_MIN
        fout.attrs["e_max_ha_atom"]  = E_MAX
        fout.attrs["energy_unit"]    = "eV (atomization)"
        fout.attrs["force_unit"]     = "eV/Ang"
        fout.attrs["coord_unit"]     = "Ang"

        for g in sorted(fin.keys()):
            grp    = fin[g]
            N      = grp["energy"].shape[0]
            stats["total_in"] += N

            numbers = grp["numbers"][:]   # (N, n_atoms)  int8
            charges = grp["charge"][:]    # (N,)          int8
            energy  = grp["energy"][:]    # (N,)          float64  Ha
            forces  = grp["forces"][:]    # (N, n_atoms, 3) float32 eV/Å

            n_atoms = int(g)

            # 1. element filter (vectorized)
            elem_mask = build_elem_mask(numbers.astype(np.int16))

            # 2. charge filter
            charge_mask = np.abs(charges) <= MAX_CHARGE

            # 3. energy/atom filter
            epa = energy / n_atoms
            energy_mask = (epa >= E_MIN) & (epa <= E_MAX)

            # 4. force filter — any atom in conformer exceeds MAX_FORCE
            fmag = np.linalg.norm(forces, axis=-1)   # (N, n_atoms)
            force_mask = fmag.max(axis=1) <= MAX_FORCE

            keep = elem_mask & charge_mask & energy_mask & force_mask
            n_keep = int(keep.sum())

            stats["dropped_elem"]   += int((~elem_mask).sum())
            stats["dropped_charge"] += int((elem_mask & ~charge_mask).sum())
            stats["dropped_energy"] += int((elem_mask & charge_mask & ~energy_mask).sum())
            stats["dropped_force"]  += int((elem_mask & charge_mask & energy_mask & ~force_mask).sum())
            stats["total_out"]      += n_keep

            print(
                f"  {g}: {N:>8,} in"
                f"  -elem {(~elem_mask).sum():>7,}"
                f"  -chg {(elem_mask & ~charge_mask).sum():>6,}"
                f"  -ene {(elem_mask & charge_mask & ~energy_mask).sum():>6,}"
                f"  -frc {(elem_mask & charge_mask & energy_mask & ~force_mask).sum():>6,}"
                f"  => {n_keep:>8,} kept"
            )

            if n_keep == 0:
                continue

            # 5. compute atomization energy (eV)
            e_atomization = compute_atomization(energy[keep], numbers[keep])

            out = fout.create_group(g)
            kw = dict(compression="gzip", compression_opts=4)

            # original datasets — sliced to kept conformers
            for key in ["coord", "forces", "numbers", "charge", "charges", "dipole", "quadrupole"]:
                out.create_dataset(key, data=grp[key][:][keep], **kw)

            # energy: keep original Ha for reference, add atomization in eV
            out.create_dataset("energy",              data=energy[keep],   **kw)
            out.create_dataset("energy_atomization",  data=e_atomization,  **kw)

    return stats


if __name__ == "__main__":
    t0 = time.time()
    print(f"Input : {INPUT}")
    print(f"Output: {OUTPUT}")
    print(f"Allowed Z : { {ELEM_SYMBOL[z]: z for z in sorted(ALLOWED_Z)} }")
    print(f"Max charge: {MAX_CHARGE}")
    print(f"Max force : {MAX_FORCE} eV/Å")
    print(f"E/atom    : [{E_MIN}, {E_MAX}] Ha\n")

    stats = process()

    total_in  = stats["total_in"]
    total_out = stats["total_out"]
    print()
    print(f"{'Input conformers':<28}: {total_in:>12,}")
    print(f"{'Dropped (non-allowed elements)':<28}: {stats['dropped_elem']:>12,}  ({100*stats['dropped_elem']/total_in:.1f}%)")
    print(f"{'Dropped (charge != 0)':<28}: {stats['dropped_charge']:>12,}  ({100*stats['dropped_charge']/total_in:.1f}%)")
    print(f"{'Dropped (energy outlier)':<28}: {stats['dropped_energy']:>12,}  ({100*stats['dropped_energy']/total_in:.1f}%)")
    print(f"{'Dropped (force > 20 eV/Å)':<28}: {stats['dropped_force']:>12,}  ({100*stats['dropped_force']/total_in:.1f}%)")
    print(f"{'Output conformers':<28}: {total_out:>12,}  ({100*total_out/total_in:.1f}%)")
    print(f"\nDone in {time.time() - t0:.1f}s")
    print(f"Written to: {OUTPUT}")
