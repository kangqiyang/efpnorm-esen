"""
Preprocess SPICE-2.0.1 wB97M-D3(BJ)/def2-TZVPPD dataset for MLFF pretraining.

Pipeline:
  1. Element filter  — keep molecules where ALL atoms are in ALLOWED_Z
                       {H, C, N, O, F, Si, P, S, Cl, Br}

  2. Charge filter   — keep conformers where sum(MBIS partial charges) ≈ 0
                       (rounded to nearest integer). Molecules without MBIS
                       charges (121 ionic Ion Pair / DES370K molecules) are
                       dropped entirely — no charge signal available.

  3. Energy filter   — drop conformers with total energy/atom outside
                       [E_MIN, E_MAX] Ha/atom. Same range as AIMNet2;
                       removes corrupted DFT entries.

  4. Force filter    — drop conformers with any |F| > MAX_FORCE eV/Å.
                       SPICE-2 stores dft_total_gradient in Ha/Bohr;
                       forces = −gradient × GRAD_TO_FORCE.

  5. Unit conversion — forces: Ha/Bohr → eV/Å; energies: Ha → eV;
                       formation_energy: Ha → eV (already atomization energy)

Output: processed/spice2_processed.h5
  Groups keyed by molecule name (same as input).
  Datasets: atomic_numbers (n_atoms,), conformations (n_keep, n_atoms, 3),
            forces (n_keep, n_atoms, 3), formation_energy (n_keep,),
            dft_total_energy (n_keep,).
  All energies in eV; forces in eV/Å; positions in Å.
"""

import h5py
import numpy as np
from pathlib import Path
import time

# ── Paths ────────────────────────────────────────────────────────────────
INPUT  = Path("raw/spice2/SPICE-2.0.1.hdf5")
OUTPUT = Path("processed/spice2_processed.h5")

# ── Filter parameters ────────────────────────────────────────────────────
ALLOWED_Z  = frozenset([1, 6, 7, 8, 9, 14, 15, 16, 17, 35])
#                       H  C  N  O  F  Si   P   S  Cl  Br
MAX_CHARGE = 0
MAX_FORCE  = 20.0     # eV/Å
E_MIN      = -1000.0  # Ha/atom
E_MAX      = -0.4     # Ha/atom

# ── Unit conversions ─────────────────────────────────────────────────────
HA_TO_EV      = 27.211386
BOHR_TO_ANG   = 0.529177
GRAD_TO_FORCE = HA_TO_EV / BOHR_TO_ANG  # Ha/Bohr → eV/Å (~51.42)


def process():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_in": 0,
        "dropped_elem": 0,
        "dropped_no_mbis": 0,
        "dropped_charge": 0,
        "dropped_energy": 0,
        "dropped_force": 0,
        "total_out": 0,
    }

    with h5py.File(INPUT, "r", locking=False) as fin, \
         h5py.File(OUTPUT, "w", locking=False) as fout:
        fout.attrs["source"]         = str(INPUT)
        fout.attrs["allowed_Z"]      = sorted(ALLOWED_Z)
        fout.attrs["max_charge"]     = MAX_CHARGE
        fout.attrs["max_force_evA"]  = MAX_FORCE
        fout.attrs["e_min_ha_atom"]  = E_MIN
        fout.attrs["e_max_ha_atom"]  = E_MAX
        fout.attrs["energy_unit"]    = "eV"
        fout.attrs["force_unit"]     = "eV/Ang"
        fout.attrs["coord_unit"]     = "Ang"

        mols = sorted(fin.keys())
        for i_mol, mol in enumerate(mols):
            g       = fin[mol]
            zs      = g["atomic_numbers"][:].astype(np.int16)   # (n_atoms,)
            n_atoms = len(zs)
            n_conf  = g["dft_total_energy"].shape[0]
            stats["total_in"] += n_conf

            # 1. element filter (molecule-level — all atoms must be in ALLOWED_Z)
            if not np.isin(zs, sorted(ALLOWED_Z)).all():
                stats["dropped_elem"] += n_conf
                continue

            # 2. charge filter — MBIS partial charges; drop molecules without MBIS
            if "mbis_charges" not in g:
                stats["dropped_no_mbis"] += n_conf
                continue
            mbis    = g["mbis_charges"][:].reshape(n_conf, n_atoms)  # (n_conf, n_atoms)
            q_total = np.round(mbis.sum(axis=1)).astype(int)          # (n_conf,)
            charge_mask = q_total == MAX_CHARGE

            # 3. energy/atom filter (total DFT energy in Ha)
            energy_ha = g["dft_total_energy"][:]   # (n_conf,) Ha
            epa       = energy_ha / n_atoms
            energy_mask = (epa >= E_MIN) & (epa <= E_MAX)

            # 4. force filter — gradient stored as Ha/Bohr, convert to eV/Å
            grad   = g["dft_total_gradient"][:].reshape(n_conf, n_atoms, 3)  # Ha/Bohr
            forces = -grad * GRAD_TO_FORCE                                     # eV/Å
            fmag_max   = np.linalg.norm(forces, axis=-1).max(axis=1)          # (n_conf,)
            force_mask = fmag_max <= MAX_FORCE

            keep   = charge_mask & energy_mask & force_mask
            n_keep = int(keep.sum())

            stats["dropped_charge"] += int((~charge_mask).sum())
            stats["dropped_energy"] += int((charge_mask & ~energy_mask).sum())
            stats["dropped_force"]  += int((charge_mask & energy_mask & ~force_mask).sum())
            stats["total_out"]      += n_keep

            if (i_mol + 1) % 10000 == 0:
                print(f"  [{i_mol+1:>6}/{len(mols):>6}]  {mol:<40}  "
                      f"{n_conf:>5} in  => {n_keep:>5} kept")

            if n_keep == 0:
                continue

            out = fout.create_group(mol)
            kw  = dict(compression="gzip", compression_opts=4)

            out.create_dataset("atomic_numbers",    data=zs.astype(np.int32), **kw)
            out.create_dataset("conformations",     data=g["conformations"][:][keep].astype(np.float32), **kw)
            out.create_dataset("forces",            data=forces[keep].astype(np.float32), **kw)
            # formation_energy: already atomization energy in Ha → convert to eV
            fe_ev = g["formation_energy"][:][keep] * HA_TO_EV
            out.create_dataset("formation_energy",  data=fe_ev.astype(np.float64), **kw)
            # total DFT energy in eV
            out.create_dataset("dft_total_energy",  data=(energy_ha[keep] * HA_TO_EV).astype(np.float64), **kw)

    return stats


if __name__ == "__main__":
    t0 = time.time()
    print(f"Input : {INPUT}")
    print(f"Output: {OUTPUT}")
    print(f"Max force : {MAX_FORCE} eV/Å")
    print(f"E/atom    : [{E_MIN}, {E_MAX}] Ha\n")

    stats = process()

    total_in  = stats["total_in"]
    total_out = stats["total_out"]
    print()
    print(f"{'Input conformers':<32}: {total_in:>12,}")
    print(f"{'Dropped (non-allowed elements)':<32}: {stats['dropped_elem']:>12,}  ({100*stats['dropped_elem']/total_in:.1f}%)")
    print(f"{'Dropped (no MBIS charges)':<32}: {stats['dropped_no_mbis']:>12,}  ({100*stats['dropped_no_mbis']/total_in:.1f}%)")
    print(f"{'Dropped (charge != 0)':<32}: {stats['dropped_charge']:>12,}  ({100*stats['dropped_charge']/total_in:.1f}%)")
    print(f"{'Dropped (energy outlier)':<32}: {stats['dropped_energy']:>12,}  ({100*stats['dropped_energy']/total_in:.1f}%)")
    print(f"{'Dropped (force > 20 eV/Å)':<32}: {stats['dropped_force']:>12,}  ({100*stats['dropped_force']/total_in:.1f}%)")
    print(f"{'Output conformers':<32}: {total_out:>12,}  ({100*total_out/total_in:.1f}%)")
    print(f"\nDone in {time.time() - t0:.1f}s")
    print(f"Written to: {OUTPUT}")
