# eafm-polyget: Data Preprocessing for EAFM-PolyGET

Preprocessing pipeline for the five datasets used in EAFM (Equilibrium-Aware Force Matching)
pretraining of PolyGET.

---

## Data location

Raw data and processed outputs are **NOT on this repo** (too large). Everything lives on the a100cse
cluster under Kangqi's account:

```
/localscratch/kyang394/mlff/eafm-polyget/data/
├── raw/                          # original downloaded files (~63 GB total)
│   ├── aimnet2/
│   │   └── aimnet2_wb97m.h5                          (14 GB)
│   ├── spice2/
│   │   └── SPICE-2.0.1.hdf5                          (35 GB)
│   ├── ani2x/
│   │   └── final_h5/ANI-2x-wB97X-631Gd.h5           (6.2 GB)
│   ├── spf/
│   │   └── solvated_protein_fragments.npz             (1.4 GB)
│   └── qdpi/
│       └── QDpiDataset-main/data/neutral/             (2.0 GB, 6 HDF5 files)
│           ├── ani.hdf5, comp6.hdf5, freesolvmd.hdf5
│           ├── geom.hdf5, re.hdf5, remd.hdf5
└── processed/                    # output of process_*.py scripts (~7 GB total)
    ├── aimnet2_processed.h5      (3.1 GB)
    ├── spice2_processed.h5       (2.6 GB)
    ├── spf_processed.h5          (870 MB)
    ├── qdpi_processed.h5         (457 MB)
    └── ani2x_processed.h5        (225 MB)
```

---

## Datasets

| Dataset | Tier | Theory | Raw frames | Post-filter | Output file |
|---------|------|--------|-----------|-------------|-------------|
| AIMNet2 | T1 | wB97M-D3(BJ)/def2-TZVPP | 5.8 M | ~4.5 M | `aimnet2_processed.h5` |
| SPICE-2 | T1 | wB97M-D3(BJ)/def2-TZVPPD | 2.0 M | ~1.76 M | `spice2_processed.h5` |
| QDpi | T1 | wB97M-D3(BJ)/def2-TZVPPD | 540 K | ~529 K | `qdpi_processed.h5` |
| SPF | T2 | revPBE-D3(BJ)/def2-TZVP | 2.73 M | ~1.75 M | `spf_processed.h5` |
| ANI-2x | T2 | wB97X/6-31G(d) | 9.65 M | ~542 K (capped) | `ani2x_processed.h5` |

Filtering applied uniformly across all datasets: element filter `{H,C,N,O,F,Si,P,S,Cl,Br}`,
neutral-only (charge = 0), energy/atom sanity bounds, force cutoff 20 eV/Å.
ANI-2x also applies a **100-frame cap per unique molecular composition** to prevent
over-represented small molecules from dominating.

---

## Processing scripts

Run each from the `data/` directory. All scripts write to `data/processed/`.

```bash
cd /nethome/kyang394/scratch/mlff/eafm-polyget/data

python process_aimnet2.py   # ~30 min
python process_spice2.py    # ~45 min
python process_qdpi.py      # ~15 min
python process_spf.py       # ~20 min
python process_ani2x.py     # ~20 min  (includes composition-cap subsampling)
```

All scripts require `h5py` and `numpy`. Use `locking=False` (already set in the code)
since `nethome` is NFS and doesn't support HDF5 POSIX locking.

---

## Output schema

Each processed HDF5 follows the same per-dataset-adapted schema:

**AIMNet2 / ANI-2x** — groups keyed by atom count string (`'007'`, `'023'`, …):
```
{atom_count}/
  numbers / species     (n_atoms,)              int32    atomic numbers
  coord / coordinates   (n_frames, n_atoms, 3)  float32  positions, Å
  forces                (n_frames, n_atoms, 3)  float32  forces, eV/Å
  energy                (n_frames,)             float64  total energy, Ha
  energy_atomization    (n_frames,)             float64  atomization energy, eV
```

**SPICE-2** — groups keyed by molecule name:
```
{molecule_name}/
  atomic_numbers        (n_atoms,)              int32    atomic numbers
  conformations         (n_frames, n_atoms, 3)  float32  positions, Å
  forces                (n_frames, n_atoms, 3)  float32  forces, eV/Å  (from −∇E)
  formation_energy      (n_frames,)             float64  atomization energy, eV
  dft_total_energy      (n_frames,)             float64  total DFT energy, eV
```

**SPF** — groups keyed by Hill-formula composition string (`'C4H8N2O2S1'`, …):
```
{composition}/
  numbers               (n_atoms,)              int32    atomic numbers
  pos                   (n_frames, n_atoms, 3)  float32  positions, Å
  forces                (n_frames, n_atoms, 3)  float32  forces, eV/Å
  energy                (n_frames,)             float32  atomization energy, eV
```

**QDpi** — groups keyed as `{source_file}/{molecule_name}` (`'ani/C2H5NO'`, …):
```
{source}/{molecule}/
  numbers               (n_atoms,)              int32    atomic numbers
  pos                   (n_frames, n_atoms, 3)  float32  positions, Å
  forces                (n_frames, n_atoms, 3)  float32  forces, eV/Å
  energy                (n_frames,)             float64  total DFT energy, eV
  energy_atomization    (n_frames,)             float64  atomization energy, eV
```

## Inspection notebook

`data/raw_data_inspection.ipynb` — full per-dataset quality analysis:
element distributions, charge histograms, force magnitude CDFs, energy outlier checks,
filter retention estimates, and atomization energy spot-checks. Read this before
modifying any filter thresholds.
