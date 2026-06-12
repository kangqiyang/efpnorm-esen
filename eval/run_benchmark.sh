#!/usr/bin/env bash
# NVE benchmark: 7 molecules × 3 seeds × 2 models = 42 trajectories.
# Run from the project root after both epoch-30 checkpoints are ready.
#
# Serial (safe, one GPU):
#   bash eval/run_benchmark.sh
#
# Parallel (two GPUs, ~2× faster):
#   MODELS=efp bash eval/run_benchmark.sh &
#   MODELS=rms bash eval/run_benchmark.sh &
#
# Override steps:  STEPS=20000 bash eval/run_benchmark.sh

set -euo pipefail

PYTHON=${PYTHON:-/nethome/kyang394/scratch/envs/MLFF/bin/python}
EFP=train/checkpoints/qdpi_L4C128_efpnorm_lr4e-4
RMS=train/checkpoints/qdpi_L4C128_rmsnorm_lr4e-4
STEPS=${STEPS:-10000}
MODELS=${MODELS:-both}   # efp | rms | both

# 7 benchmark molecules (moderate force 1.0–2.4 eV/Å, 9–22 atoms)
# idx    formula     n_atoms  max_F
# 23370  C2H3NOS2     9       1.002
# 23355  C2H2N4O2    10       1.003
# 20827  C5H8S2      15       1.483
# 14549  C6H10O      17       1.680
#  9176  C8H13N      22       1.852
# 21966  C5H10O2     17       2.208
#  8751  C8H10N4     22       2.396
INDICES=(23370 14549 8751)   # small/medium/medium — expand to all 7 after sanity check
SEEDS=(42 0 1)

run_one() {
    local ckpt=$1 idx=$2 seed=$3
    echo ">>> $(basename $ckpt)  idx=$idx  seed=$seed"
    $PYTHON eval/run_md.py \
        --checkpoint_dir "$ckpt" \
        --sample_idx "$idx" \
        --steps "$STEPS" \
        --seed "$seed"
}

for idx in "${INDICES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        [[ $MODELS == "efp"  || $MODELS == "both" ]] && run_one "$EFP" "$idx" "$seed"
        [[ $MODELS == "rms"  || $MODELS == "both" ]] && run_one "$RMS" "$idx" "$seed"
    done
done

echo "=== benchmark complete ==="
