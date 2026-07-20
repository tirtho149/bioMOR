#!/usr/bin/env bash
#SBATCH --job-name=baron-bio-ablation
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --array=0-4%5
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/baron-bio-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/baron-bio-%A_%a.out
# EXPLORATORY (parallel): Baron (single-cell) under each biology row of BIOLOGY_SC_VS_PATHWAY.md
# that IS applicable to single-cell, on the shared 5-fold folds (cv.cv_folds, seed 42). One mode
# per array task -> runs in parallel.
#   0 none         : no biology (baseline row)
#   1 learned_bio  : EMBEDDING site only (learned graph warm-started from co-expression)
#   2 route_graph  : ROUTER site only (zero-init graph-conv over the learned gene graph)
#   3 bio_both     : BOTH sites (canonical single-cell bioMoR)
#   4 learned_fused: fixed-graph FUSION analogue (learned graph fused with a co-expression graph)
# NOTE: the two multi-omics-ONLY rows -- pathway attention bias and pathway fixed-graph fusion --
# are gated to marker_mode='pathway' (model.py:167-168, 63) and CANNOT run on single-cell Baron;
# that gating is the MD's central point (single-cell genomap features are anonymized -> no
# provided/named gene graph). learned_fused above is the closest single-cell analogue.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
cd "$PROJ"
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

MODES=(none learned_bio route_graph bio_both learned_fused)
M="${MODES[$SLURM_ARRAY_TASK_ID]}"
OUT="$PROJ/results/cv5/baron_biology_ablation"

echo "[baron-bio] $(date) mode=$M host=$(hostname)"; nvidia-smi -L || true
python -m recursive_marker_transformer.bio_learned_genomap \
    --dataset Baron --modes "$M" --cv_folds 5 --K 4 --epochs 100 --patience 15 \
    --n_markers 128 --seed 42 --device cuda --out "$OUT"
echo "[baron-bio] DONE rc=$? mode=$M $(date)"
