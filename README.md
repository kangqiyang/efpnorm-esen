# efpnorm-esen

Investigating whether **EFPNorm** (Equivariant Force-Preserving Normalization) improves
MD stability over the RMSNorm baseline in eSEN-style MLFFs, beyond what force MAE alone captures.

---

## Hypothesis

RMSNorm's `1/sqrt(rms² + ε)` scale factor diverges when activations are near zero, making
the Jacobian rank-deficient and destroying force gradients in the backward pass.
EFPNorm replaces `ε` with a learnable `c² = softplus(log_c_raw)²`, keeping the scale bounded
away from zero. The prediction: deeper networks trained with EFPNorm should produce smoother
potential energy surfaces and better NVE energy conservation.

---

## Model

We train **eSCNMDBackbone + MLP_EFS_Head** from scratch, matching the official eSEN-SM
conserving architecture exactly (verified by inspecting `esen_sm_conserving_all.pt`).

### Architecture (confirmed against official eSEN-SM conserving checkpoint)

| Parameter | Value |
|-----------|-------|
| `num_layers` | 4 |
| `lmax` / `mmax` | 2 / 2 |
| `sphere_channels` | 128 |
| `hidden_channels` | 128 |
| `edge_channels` | 128 |
| `num_distance_basis` | 64 |
| `distance_function` | gaussian |
| `ff_type` | spectral |
| `norm_type` | rms_norm_sh → **replaced by EFPNorm** |
| `act_type` | gate |
| `chg_spin_emb_type` | rand_emb |
| `cs_emb_grad` | True |
| `direct_forces` | False (conservative: F = −∇E via autograd) |
| `cutoff` | 6.0 Å |
| `max_num_elements` | 100 |
| Total params | **6.3M** |

EFPNorm replaces 9 `EquivariantRMSNormArraySphericalHarmonicsV2` layers (~2.1 per block).

### EFPNorm

```
RMSNorm : scale = 1 / sqrt(rms² + ε)        ε = 1e-5  (near-zero → large scale)
EFPNorm : scale = 1 / sqrt(rms² + c²)        c = softplus(log_c_raw) ≈ 1.0  (bounded)
```

`log_c_raw` is a scalar learnable parameter initialised so `c ≈ 1.0`.
Affine weights are copied from the RMSNorm layer being replaced, so output is
identical at init; the two paths diverge only as `c` is learned.

---

## Training

### Config

| Setting | Value |
|---------|-------|
| Dataset | QDPi (wB97M-D3/def2-TZVPPD) |
| Split | train / val |
| Max atoms | 100 |
| Batch size | 8 |
| Optimizer | AdamW, weight_decay=1e-3 |
| LR | 4e-4 → cosine decay → 4e-6 |
| Epochs | 10 |
| Loss | L1 force (coef=1.0) + L1 energy/atom (coef=0.01) |
| Gradient clip | 100.0 |
| Seed | 42 |

### Commands

```bash
# EFPNorm
python train/pretrain.py --dataset qdpi --epochs 10 --lr 4e-4

# RMSNorm baseline
python train/pretrain.py --dataset qdpi --epochs 10 --lr 4e-4 --no_efp_norm
```

---

## Preliminary Results

### Force MAE — QDPi val split (meV/Å)

| Epoch | EFPNorm | RMSNorm |
|-------|--------:|--------:|
| 1  | 42.9 | 43.4 |
| 2  | 30.6 | 39.1 |
| 3  | 31.4 | 28.5 |
| 4  | 23.5 | 29.1 |
| 5  | 21.6 | 21.7 |
| 6  | 16.9 | 17.4 |
| 7  | 15.7 | 15.1 |
| 8  | 14.0 | 13.2 |
| 9  | 12.6 | 12.2 |
| **10** | **11.6** | **11.6** |

Training time: ~25.9 h per run (10 epochs on A100).
Best val F-MAE: EFPNorm **11.645 meV/Å**, RMSNorm **11.563 meV/Å** — statistically indistinguishable.

### NVE MD Stability — QDPi val, sample 0 (C₄H₇N₃S, 15 atoms, 300 K, 0.5 fs)

| Metric | EFPNorm | RMSNorm |
|--------|--------:|--------:|
| Energy drift / atom | 2.67 meV | 2.67 meV |
| Max temperature | 1581 K | 1588 K |
| Steps completed | 50 / 50 | 50 / 50 |
| NaN / explosion | No | No |

> **Both metrics are essentially identical at L4.**
> The EFPNorm hypothesis is likely correct but requires deeper networks to manifest.
> At L4 there are only 9 norm layers; gradient corruption is limited.
> Next: L8 (17 norms, ~12.4M params) and L12 (25 norms, ~18.5M params).

---

## Evaluation

### MD pipeline

```bash
# Run NVE for 1 ps (2000 × 0.5 fs)
python eval/run_md.py --checkpoint_dir train/checkpoints/qdpi_L4C128_efpnorm_lr4e-4
python eval/run_md.py --checkpoint_dir train/checkpoints/qdpi_L4C128_rmsnorm_lr4e-4
```

Outputs written to `eval/md_runs/<tag>/`:

| File | Contents |
|------|----------|
| `md_metrics.csv` | step, epot, ekin, etot, temp, max_force, drift_per_atom |
| `trajectory.traj` | ASE binary trajectory |
| `summary.json` | energy_drift_per_atom, failed flag, n_model_calls |

Key CLI args: `--sample_idx`, `--temperature`, `--timestep_fs`, `--steps`, `--device`, `--seed`.

---

## Project structure

```
efpnorm-esen/
├── model/
│   └── efpnorm.py              # EFPNorm + EquivariantEFPNorm
├── data/
│   ├── process_*.py            # per-dataset preprocessing
│   ├── convert_to_unified.py   # normalise to common H5 schema
│   └── h5_to_asedb.py          # convert unified H5 → ASE SQLite DB
├── train/
│   └── pretrain.py             # from-scratch training, EFPNorm/RMSNorm switch
└── eval/
    └── run_md.py               # NVE MD evaluation + energy drift metric
```

---

## Data

Raw data and checkpoints are not in this repo. Everything lives on the cluster:

```
/localscratch/kyang394/mlff/efpnorm-esen/
├── data/raw/qdpi/              # QDPi source (~2.0 GB)
├── data/asedb/                 # qdpi_train.db, qdpi_val.db, ...
└── train/checkpoints/
    ├── qdpi_L4C128_efpnorm_lr4e-4/
    └── qdpi_L4C128_rmsnorm_lr4e-4/
```

### Datasets

| Dataset | Theory | Frames |
|---------|--------|--------|
| QDPi | wB97M-D3(BJ)/def2-TZVPPD | ~529 K |
| AIMNet2 | wB97M-D3(BJ)/def2-TZVPP | ~4.5 M |
| SPICE-2 | wB97M-D3(BJ)/def2-TZVPPD | ~1.76 M |
| ANI-2x | wB97X/6-31G(d) | ~542 K |
| SPF | revPBE-D3(BJ)/def2-TZVP | ~1.75 M |

Energies stored as atomization energies (eV); no reference subtraction needed at training time.

### Preprocessing

```bash
python data/process_qdpi.py
python data/convert_to_unified.py --dataset qdpi
python data/h5_to_asedb.py --dataset qdpi
```
