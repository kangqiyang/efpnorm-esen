"""
Preprocess SPF (Solvated Protein Fragments) revPBE-D3(BJ)/def2-TZVP dataset
for MLFF pretraining.

Source: single .npz with flat arrays, all frames zero-padded to max 120 atoms.

  | Key | Shape            | Dtype   | Unit / Notes                        |
  |-----|------------------|---------|-------------------------------------|
  | R   | (2731180, 120,3) | float32 | Å — positions (zero-padded)         |
  | F   | (2731180, 120,3) | float32 | eV/Å — forces (zero-padded)         |
  | E   | (2731180,)       | float32 | eV — atomization energy (pre-built) |
  | Z   | (2731180, 120)   | int32   | atomic numbers (zero-padded)        |
  | N   | (2731180,)       | int32   | real atom count per frame           |
  | Q   | (2731180,)       | float32 | total molecular charge              |

Pipeline:
  1. Element filter  — trivially passes: SPF contains only {H, C, N, O, S}.

  2. Charge filter   — keep frames with round(Q) == 0 (neutral only).
                       ~36% of frames are charged and dropped.

  3. Energy filter   — drop frames with formation energy/atom outside
                       [E_MIN_EV, E_MAX_EV] eV/atom. Energy is already
                       atomization energy in eV, so different thresholds apply.

  4. Force filter    — drop frames with any |F| > MAX_FORCE eV/Å.
                       Forces are already in eV/Å (no conversion needed).

Output: processed/spf_processed.h5
  Groups keyed by atomic composition string (e.g. "C4H8N2O2S1"), so all
  frames in a group share the same molecule type. Groups allow clean
  conformer-stacking and are compatible with Rui's pretraining HDF5 schema.

  Datasets per group:
    numbers  (n_atoms,)              int32   atomic numbers
    pos      (n_frames, n_atoms, 3)  float32 positions in Å
    forces   (n_frames, n_atoms, 3)  float32 forces in eV/Å
    energy   (n_frames,)             float32 atomization energy in eV

Memory: processes in chunks to avoid loading ~8 GB at once. Uses mmap for R/F/Z.
"""

import h5py
import numpy as np
from collections import defaultdict
from pathlib import Path
import time

# ── Paths ────────────────────────────────────────────────────────────────
INPUT  = Path("raw/spf/solvated_protein_fragments.npz")
OUTPUT = Path("processed/spf_processed.h5")

# ── Filter parameters ────────────────────────────────────────────────────
MAX_FORCE   = 20.0   # eV/Å
E_MIN_EV    = -10.0  # eV/atom  (formation energy already in eV)
E_MAX_EV    =  0.5   # eV/atom

CHUNK_SIZE  = 50_000  # frames per chunk when scanning for force filter

# ── Element symbols (for composition key construction) ───────────────────
ELEM_SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O", 16: "S"}


def composition_key(z_row: np.ndarray, n: int) -> str:
    """Build a composition string from atomic numbers, e.g. 'C4H8N2O2S1'."""
    from collections import Counter
    counts = Counter(int(z) for z in z_row[:n])
    # canonical order: C, H, N, O, S (Hill order for organics)
    hill = [6, 1, 7, 8, 16]
    parts = []
    for z in hill:
        if counts[z] > 0:
            sym = ELEM_SYMBOL.get(z, f"Z{z}")
            parts.append(f"{sym}{counts[z]}")
    return "".join(parts)


def process():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # ── Load small arrays fully; mmap large arrays ───────────────────────
    data  = np.load(INPUT, mmap_mode="r")
    N_arr = np.array(data["N"])   # (n_frames,)  int32  — small, load fully
    Q_arr = np.array(data["Q"])   # (n_frames,)  float32
    E_arr = np.array(data["E"])   # (n_frames,)  float32
    Z_arr = data["Z"]             # mmap  (n_frames, 120) int32
    R_arr = data["R"]             # mmap  (n_frames, 120, 3) float32
    F_arr = data["F"]             # mmap  (n_frames, 120, 3) float32
    n_total = len(N_arr)

    # ── Pass 1: build mask using N, Q, E (vectorized) ───────────────────
    neutral_mask = (np.round(Q_arr).astype(int) == 0)
    epa          = E_arr / N_arr
    energy_mask  = (epa >= E_MIN_EV) & (epa <= E_MAX_EV)
    pre_mask     = neutral_mask & energy_mask

    # ── Pass 2: force filter in chunks (reads F lazily) ──────────────────
    force_ok = np.zeros(n_total, dtype=bool)
    pre_idx  = np.where(pre_mask)[0]

    # process in sorted chunks for sequential mmap reads
    for start in range(0, len(pre_idx), CHUNK_SIZE):
        chunk_idx = pre_idx[start : start + CHUNK_SIZE]
        F_chunk   = F_arr[chunk_idx]          # (chunk, 120, 3) float32
        # max |F| per frame over all positions (including zero-padded atoms,
        # which have F=0 and do not affect the max if real forces are > 0)
        fmag_max  = np.linalg.norm(F_chunk, axis=-1).max(axis=1)
        force_ok[chunk_idx] = fmag_max <= MAX_FORCE

        if (start // CHUNK_SIZE + 1) % 10 == 0:
            done = min(start + CHUNK_SIZE, len(pre_idx))
            print(f"  Force scan: {done:>8,}/{len(pre_idx):>8,} pre-filtered frames")

    keep_mask = pre_mask & force_ok
    keep_idx  = np.where(keep_mask)[0]

    # ── Pass 3: group kept frames by composition ─────────────────────────
    print(f"\nGrouping {len(keep_idx):,} kept frames by composition...")
    comp_to_idx: dict[str, list[int]] = defaultdict(list)
    for idx in keep_idx:
        key = composition_key(Z_arr[idx], int(N_arr[idx]))
        comp_to_idx[key].append(idx)

    # ── Pass 4: write each composition group ─────────────────────────────
    total_out   = 0
    n_dropped_e = int((~neutral_mask).sum())
    n_dropped_c = int((neutral_mask & ~energy_mask).sum())
    n_dropped_f = int((pre_mask & ~force_ok).sum())

    with h5py.File(OUTPUT, "w", locking=False) as fout:
        fout.attrs["source"]         = str(INPUT)
        fout.attrs["max_force_evA"]  = MAX_FORCE
        fout.attrs["e_min_ev_atom"]  = E_MIN_EV
        fout.attrs["e_max_ev_atom"]  = E_MAX_EV
        fout.attrs["energy_unit"]    = "eV (atomization)"
        fout.attrs["force_unit"]     = "eV/Ang"
        fout.attrs["coord_unit"]     = "Ang"

        kw = dict(compression="gzip", compression_opts=4)
        for comp_key in sorted(comp_to_idx.keys()):
            indices  = np.array(sorted(comp_to_idx[comp_key]), dtype=np.int64)
            n_frames = len(indices)
            n_atoms  = int(N_arr[indices[0]])  # all frames in group have same n_atoms

            # read actual (unpadded) data
            pos_out = R_arr[indices, :n_atoms, :]   # (n_frames, n_atoms, 3)
            frc_out = F_arr[indices, :n_atoms, :]   # (n_frames, n_atoms, 3)
            ene_out = E_arr[indices]                 # (n_frames,)
            z_out   = Z_arr[indices[0], :n_atoms]   # (n_atoms,) — same for whole group

            out = fout.create_group(comp_key)
            out.create_dataset("numbers", data=z_out.astype(np.int32),           **kw)
            out.create_dataset("pos",     data=pos_out.astype(np.float32),        **kw)
            out.create_dataset("forces",  data=frc_out.astype(np.float32),        **kw)
            out.create_dataset("energy",  data=ene_out.astype(np.float32),        **kw)

            total_out += n_frames

    return {
        "total_in":       n_total,
        "dropped_charge": n_dropped_e,
        "dropped_energy": n_dropped_c,
        "dropped_force":  n_dropped_f,
        "total_out":      total_out,
    }


if __name__ == "__main__":
    t0 = time.time()
    print(f"Input : {INPUT}")
    print(f"Output: {OUTPUT}")
    print(f"Max force : {MAX_FORCE} eV/Å")
    print(f"E/atom    : [{E_MIN_EV}, {E_MAX_EV}] eV\n")

    stats = process()

    total_in  = stats["total_in"]
    total_out = stats["total_out"]
    print()
    print(f"{'Input frames':<32}: {total_in:>12,}")
    print(f"{'Dropped (charge != 0)':<32}: {stats['dropped_charge']:>12,}  ({100*stats['dropped_charge']/total_in:.1f}%)")
    print(f"{'Dropped (energy outlier)':<32}: {stats['dropped_energy']:>12,}  ({100*stats['dropped_energy']/total_in:.1f}%)")
    print(f"{'Dropped (force > 20 eV/Å)':<32}: {stats['dropped_force']:>12,}  ({100*stats['dropped_force']/total_in:.1f}%)")
    print(f"{'Output frames':<32}: {total_out:>12,}  ({100*total_out/total_in:.1f}%)")
    print(f"\nDone in {time.time() - t0:.1f}s")
    print(f"Written to: {OUTPUT}")
