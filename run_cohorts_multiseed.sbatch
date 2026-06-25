#!/usr/bin/env bash
#SBATCH --job-name=smart-coh-ms
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --array=0-4
#SBATCH --mail-user=tirtho@iastate.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/coh-ms-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/coh-ms-%A_%a.out
# =============================================================================
# Multi-seed cohort-wise modeling. One array task per seed (0..4). Each seed
# runs all THREE model variants x 4 cohorts x 5 clinical tasks, so Table 4 can
# report mean +/- std over 5 seeds per (cohort x task x model).
# Output: results_cohorts_ms/<variant>/seed<S>/<cohort>/<task>.json
#   sbatch run_cohorts_multiseed.sbatch
# =============================================================================
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
SEED=$SLURM_ARRAY_TASK_ID

echo "[coh-ms] seed=$SEED host=$(hostname) date=$(date)"
nvidia-smi -L

COMMON="--csv data/tcga/unified_bio5.csv --epochs 40 --d_model 128 --n_markers 256 \
        --batch_size 32 --lr 3e-4 --patience 8 --device cuda --seed $SEED \
        --tasks pathologic_stage pathologic_T pathologic_N os_binary tumor_status"

for COH in breast lung thyroid head_neck; do
  $PY -m recursive_marker_transformer.genonet_tasks $COMMON --cohort "$COH" \
      --recursion_mode expert \
      --out "results_cohorts_ms/shared/seed$SEED/$COH"
  echo "[coh-ms] s$SEED shared $COH rc=$? date=$(date)"

  $PY -m recursive_marker_transformer.genonet_tasks $COMMON --cohort "$COH" \
      --recursion_mode expert --no_share_weights \
      --out "results_cohorts_ms/independent/seed$SEED/$COH"
  echo "[coh-ms] s$SEED indep $COH rc=$? date=$(date)"

  $PY -m recursive_marker_transformer.genonet_tasks $COMMON --cohort "$COH" \
      --recursion_mode token \
      --out "results_cohorts_ms/mor/seed$SEED/$COH"
  echo "[coh-ms] s$SEED mor $COH rc=$? date=$(date)"
done

echo "[coh-ms] seed=$SEED DONE date=$(date)"
