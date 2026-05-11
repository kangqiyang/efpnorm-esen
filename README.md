# eafm-polyget

Machine learning force field for polymer chemistry, built on an equivariant L=1 message-passing architecture.

---

## Project structure

```
eafm-polyget/
├── data/                         # preprocessing pipeline
│   ├── process_aimnet2.py
│   ├── process_spice2.py
│   ├── process_qdpi.py
│   ├── process_spf.py
│   ├── process_ani2x.py
│   ├── convert_to_unified.py     # normalise all five datasets to a common schema
│   └── raw_data_inspection.ipynb # per-dataset quality analysis
├── model/
│   ├── efpnorm.py                # EFPNorm: force-preserving normalisation
│   ├── message_passing.py        # equivariant message passing block
│   └── polyget.py                # full model
└── train/
    ├── loss.py                   # force MAE / MSE losses
    ├── optimizer.py              # AdamW with 5x LR for vector parameters
    └── trainer.py                # DDP training loop
```

---

## Data

Raw data and processed outputs are not in this repo (too large). Everything lives on the a100cse cluster:

```
/localscratch/kyang394/mlff/eafm-polyget/data/
├── raw/                          # original downloaded files (~63 GB)
│   ├── aimnet2/aimnet2_wb97m.h5                        (14 GB)
│   ├── spice2/SPICE-2.0.1.hdf5                         (35 GB)
│   ├── ani2x/final_h5/ANI-2x-wB97X-631Gd.h5           (6.2 GB)
│   ├── spf/solvated_protein_fragments.npz               (1.4 GB)
│   └── qdpi/QDpiDataset-main/data/neutral/             (2.0 GB)
├── processed/                    # output of process_*.py (~7 GB)
└── unified/                      # output of convert_to_unified.py
```

### Datasets

| Dataset | Tier | Theory | Raw frames | Post-filter |
|---------|------|--------|-----------|-------------|
| AIMNet2 | T1 | wB97M-D3(BJ)/def2-TZVPP | 5.8 M | ~4.5 M |
| SPICE-2 | T1 | wB97M-D3(BJ)/def2-TZVPPD | 2.0 M | ~1.76 M |
| QDpi | T1 | wB97M-D3(BJ)/def2-TZVPPD | 540 K | ~529 K |
| SPF | T2 | revPBE-D3(BJ)/def2-TZVP | 2.73 M | ~1.75 M |
| ANI-2x | T2 | wB97X/6-31G(d) | 9.65 M | ~542 K (capped) |

Filters applied across all datasets: elements `{H,C,N,O,F,Si,P,S,Cl,Br}`, neutral only, energy/atom sanity bounds, max force 20 eV/Å. ANI-2x also applies a 100-frame cap per unique molecular composition.

### Preprocessing

Run from the `data/` directory:

```bash
cd /nethome/kyang394/scratch/mlff/eafm-polyget/data

python process_aimnet2.py   # ~30 min
python process_spice2.py    # ~45 min
python process_qdpi.py      # ~15 min
python process_spf.py       # ~20 min
python process_ani2x.py     # ~20 min
python convert_to_unified.py
```

Requires `h5py` and `numpy`. Use `locking=False` (already set) — nethome is NFS and does not support HDF5 POSIX locking.

---

## Model

PolyGET is a PaiNN-style L=1 equivariant force field. Key components:

- **EFPNorm** (`model/efpnorm.py`) — replaces RMSNorm with a full-rank normalisation that preserves force gradients through the autograd chain
- **VecNE** — vector neighbor encoding; unit displacement vectors injected into the vector pathway
- **vecLR 5×** (`train/optimizer.py`) — 5× learning rate for vector parameters, correcting a systematic gradient attenuation from double-backprop

Best config from ablations: L=24, hidden=128, cutoff=5 Å.

---

## Notes

- See `data/raw_data_inspection.ipynb` for full per-dataset quality analysis. Read before modifying any filter thresholds.
- Training target energies are atomization energies (eV), not raw DFT totals.
- SPF energies are already atomization energies in eV; all others require reference subtraction (done in the processing scripts).
