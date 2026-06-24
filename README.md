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

### Learned c Values (AIMNet2 best checkpoint)

![EFPNorm learned c values](eval/efpnorm_c_values.png)

Each of the 9 EFPNorm layers learns an independent `c`. Extracted from
`train/checkpoints/aimnet2_L4C128_efpnorm_lr4e-4/best.pt`:

| Layer | c |
|-------|--:|
| Block 0 — norm₁ (pre-attn) | 0.966 |
| Block 0 — norm₂ (pre-FFN)  | 0.204 |
| Block 1 — norm₁ (pre-attn) | **2.264** |
| Block 1 — norm₂ (pre-FFN)  | 0.063 |
| Block 2 — norm₁ (pre-attn) | 1.567 |
| Block 2 — norm₂ (pre-FFN)  | 0.042 |
| Block 3 — norm₁ (pre-attn) | 0.671 |
| Block 3 — norm₂ (pre-FFN)  | 0.175 |
| Final norm                   | 0.752 |

**Key observations:**

- **norm₁ (pre-attention) learned large c** — blocks 1–2 pushed well above the
  init of 1.0, indicating the model actively uses EFP protection before attention.
  Equivariant SH features entering attention can be sparse/near-zero, and large c
  prevents the normalization scale from diverging there.
- **norm₂ (pre-FFN) learned c ≈ 0.04–0.20** — all four blocks collapsed toward
  near-standard RMSNorm. EFP protection is essentially unused before the FFN,
  suggesting FFN inputs are already well-distributed.
- **Protection concentrates in middle blocks** — Block 1 norm₁ has the highest c
  (2.26); the first and last blocks are closer to init or below. The gradient
  instability the hypothesis targets manifests most strongly in middle-depth
  pre-attention norms.
- **Implication** — a *selective* EFPNorm (only norm₁ in blocks 1–2, RMSNorm
  elsewhere) may capture most of the tail-stability benefit with fewer parameters.

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

### NVE MD Stability — QDPi val (2.5 ps, 300 K, 0.5 fs, seed 42)

We ran a systematic comparison across 80 val-set molecules split into two conditions
based on their DFT reference max force norm (`eval/rank_md_candidates.py`):

| Condition | max_F range | n molecules | EFP wins | RMS wins |
|-----------|-------------|:-----------:|:--------:|:--------:|
| Equilibrated | < 0.15 eV/Å | 30 | 10 (33%) | **20 (67%)** |
| Strained | ~19–20 eV/Å | 46 valid / 50 | 32 (64%) | 18 (36%) |

**Paired t-test results (energy drift per atom, meV/atom):**

| Condition | EFP mean | RMS mean | Mean diff (EFP−RMS) | t | p | Cohen's d |
|-----------|:--------:|:--------:|:-------------------:|:-:|:-:|:---------:|
| Equilibrated | 0.168 | 0.164 | +0.004 | 2.76 | **0.010** | 0.50 |
| Strained | 5.744 | 5.778 | −0.034 | −0.55 | 0.585 (n.s.) | −0.08 |

**Interpretation:**

- On near-equilibrium structures, RMSNorm is statistically significantly better (p = 0.01),
  but the absolute effect is negligible (~0.004 meV/atom, ~2.7% relative difference).
- On strained structures, EFPNorm wins more often by head-count (64%) but the mean drift
  difference is only 0.034 meV/atom (~0.6%), which is not statistically significant (p = 0.59).
- **Overall: EFPNorm shows no meaningful MD stability advantage over RMSNorm at L4 depth**
  on QDPi checkpoints. The gradient-preservation argument may require deeper networks
  (more norm layers) to manifest in MD observables.
- 4 of 50 strained runs failed (temperature blow-up); excluded from t-test.

### NVE MD Stability — AIMNet2 val (2.5 ps, 300 K, 0.5 fs, seed 42)

Paired comparison across 50 randomly-selected val-set molecules, both models run from
identical initial conditions. Results saved in `eval/md_runs/comparison_aimnet2_full50.json`.

**Win count:** EFPNorm wins 31/50 molecules (62%) by lower drift per atom.

**Drift statistics (meV/atom):**

| | EFPNorm | RMSNorm |
|---|---:|---:|
| Mean (all 50) | 0.910 | 1.937 |
| Mean (stable 47) | 0.242 | 0.315 |
| Median (all 50) | 0.163 | 0.172 |
| Std | 3.69 | 6.48 |
| p90 | 0.578 | 1.207 |

**Statistical tests (paired, non-parametric):**

| Test | All 50 | Stable 47 |
|------|:------:|:---------:|
| Binomial (H₀: p=0.5) | p = 0.060 (n.s.) | p = 0.072 (n.s.) |
| Wilcoxon signed-rank | p = 0.063 (n.s.) | p = 0.076 (n.s.) |

The 31/50 win count and the 2× mean gap are both **not statistically significant at p < 0.05**.
93% of the mean gap is attributable to just 3 outlier molecules; the median difference on
stable molecules is only +0.003 meV/atom (95% bootstrap CI: [−0.003, +0.012]).

**Tail stability (threshold: 5 meV/atom):**

| Category | Count |
|----------|:-----:|
| Both stable | 47/50 |
| Only RMSNorm explodes | 1/50 |
| Both explode | 2/50 |
| Only EFPNorm explodes | 0/50 |

EFPNorm has zero solo blow-ups vs. one for RMSNorm (mol 253132, C15H22O: 2.2 vs 31.2 meV/atom).
This tail observation is consistent with the hypothesis but n=50 is too small to test it
rigorously — the solo-explosion count is 1 vs 0, which gives p = 0.5 by a one-sided sign test.

**Interpretation:**

- On typical molecules the two models are essentially indistinguishable (median ratio 1.03,
  CI crosses 1.0).
- The mean gap is dominated by a small number of molecules where RMSNorm diverges much more
  severely; EFPNorm appears more robust in the tail, but this is not yet statistically confirmed
  at n=50.
- **To test the tail-stability hypothesis properly, ~150–200 molecules are needed** to have
  80% power to detect the observed explosion-rate difference.

### NVE MD Stability — AIMNet2 val (200-molecule extended run)

Follow-up to the 50-molecule pilot above, directly testing the tail-stability hypothesis.
Results saved in `eval/md_runs/comparison_aimnet2_full200.json`.

**Win count:** EFPNorm wins 110/200 molecules (55%) by lower drift per atom.

**Drift statistics (meV/atom):**

| | EFPNorm | RMSNorm |
|---|---:|---:|
| Mean | 3.06 | 4.36 |
| Median | 0.215 | 0.212 |
| Std | 9.18 | 21.47 |
| p90 | 2.23 | 4.09 |
| p95 | 27.9 | 27.4 |
| Max | 53.2 | 280.1 |
| Failed | 1 | 1 |

**Statistical test:**

| Test | Result |
|------|--------|
| Sign test (efp wins=110, rms wins=89) | z = 1.49, p ≈ 0.14 (n.s.) |

**Breakdown by molecule size:**

| Atom count | n | EFP wins | EFP mean (meV/atom) | RMS mean (meV/atom) |
|------------|:-:|:--------:|--------------------:|--------------------:|
| 1–10 | 13 | 9 (69%) | 0.37 | 0.43 |
| 11–20 | 69 | 28 (41%) | 0.39 | 0.37 |
| 21–30 | 61 | 37 (61%) | 0.30 | 0.29 |
| 31–50 | 51 | 34 (67%) | 8.93 | 14.66 |
| 51+ | 6 | 2 (33%) | 17.7 | 12.6 |

**Tail stability (threshold: 5 meV/atom):**

| Category | Count |
|----------|:-----:|
| High-drift molecules (either > 5 meV/atom) | 21/200 |
| Biggest RMS failure | mol 39411 (C17H25N3): efp=40 vs rms=**280** meV/atom |
| Biggest EFP failure | mol 357430 (C13H25NO5S): efp=32 vs rms=1 meV/atom |

**Interpretation:**

- The overall 55% win rate is **not statistically significant** (p ≈ 0.14). Median difference
  is negligible (0.215 vs 0.212 meV/atom) — on typical molecules the two models are equivalent.
- The mean gap (3.1 vs 4.4 meV/atom) is almost entirely driven by tail outliers, particularly
  one catastrophic rmsnorm blow-up (C17H25N3: 280 meV/atom vs 40 for efpnorm).
- **EFPNorm's advantage concentrates in large molecules (31–50 atoms):** 34/51 wins,
  mean drift 8.9 vs 14.7 meV/atom. This is consistent with the gradient-preservation hypothesis
  mattering more when there are more norm layers being traversed in deeper/wider molecular graphs.
- Small molecules (<30 atoms, n=143): no meaningful difference.
- EFPNorm also has tail failures (5 molecules where efp is >10 meV worse than rms), so neither
  model is unconditionally more stable — efpnorm simply fails less catastrophically on average.

---

## Evaluation

### MD pipeline

```bash
# Single run
python eval/run_md.py --checkpoint_dir train/checkpoints/qdpi_L4C128_efpnorm_lr4e-4

# Paired comparison across many molecules (efpnorm vs rmsnorm in parallel)
python eval/run_md_compare.py \
    --device0 cuda:0 --device1 cuda:1 \
    --steps 5000 --out_json eval/md_runs/comparison_equil.json \
    --skip_existing \
    --sample_indices 2966 15927 9223 ...

# Rank val-set structures by DFT force norm (equilibrated / strained)
python eval/rank_md_candidates.py --dataset qdpi --split val --top_k 50
```

Outputs written to `eval/md_runs/<tag>/`:

| File | Contents |
|------|----------|
| `md_metrics.csv` | step, epot, ekin, etot, temp, max_force, drift_per_atom |
| `trajectory.traj` | ASE binary trajectory |
| `summary.json` | energy_drift_per_atom, failed flag, n_model_calls |

`run_md_compare.py` key args: `--out_json` (output path), `--skip_existing` (resume interrupted runs).

Key CLI args for `run_md.py`: `--sample_idx`, `--temperature`, `--timestep_fs`, `--steps`, `--device`, `--seed`.

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
    ├── run_md.py               # NVE MD evaluation + energy drift metric
    ├── run_md_compare.py       # paired efpnorm vs rmsnorm comparison across molecules
    └── rank_md_candidates.py   # rank val-set structures by DFT force norm
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
