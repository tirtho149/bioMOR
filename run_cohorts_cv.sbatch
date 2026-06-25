#!/usr/bin/env bash
#SBATCH --job-name=smart-coh-cv
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --array=0-2
#SBATCH --mail-user=tirtho@iastate.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/coh-cv-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/coh-cv-%A_%a.out
# =============================================================================
# Stratified 5-fold CV, cohort-wise. One array task per MODEL VARIANT (0=shared,
# 1=independent, 2=mor). Each runs unified cancer_type + the 5 clinical tasks
# within each of the 4 cohorts, reporting mean+/-std over folds (robust to the
# majority-class collapse seen on tiny imbalanced cohorts).
# Output: results_cohorts_cv/<variant>/{unified|<cohort>}/<task>.json
#   sbatch run_cohorts_cv.sbatch
# =============================================================================
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

VARIANTS=(shared independent mor)
V=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
case "$V" in
  shared)      MODEFLAGS="--recursion_mode expert" ;;
  independent) MODEFLAGS="--recursion_mode expert --no_share_weights" ;;
  mor)         MODEFLAGS="--recursion_mode token" ;;
esac

echo "[coh-cv] variant=$V host=$(hostname) date=$(date)"
nvidia-smi -L

COMMON="--csv data/tcga/unified_bio5.csv --epochs 40 --d_model 128 --n_markers 256 \
        --batch_size 32 --lr 3e-4 --patience 8 --device cuda --cv_folds 5 $MODEFLAGS"
CLIN="pathologic_stage pathologic_T pathologic_N os_binary tumor_status"

# Unified cancer_type (pooled across all cohorts), 5-fold CV
$PY -m recursive_marker_transformer.genonet_tasks $COMMON \
    --tasks cancer_type --out "results_cohorts_cv/$V/unified"
echo "[coh-cv] $V unified/cancer_type rc=$? date=$(date)"

# Per-cohort clinical/outcome tasks, 5-fold CV within each cohort
for COH in breast lung thyroid head_neck; do
  $PY -m recursive_marker_transformer.genonet_tasks $COMMON \
      --cohort "$COH" --tasks $CLIN --out "results_cohorts_cv/$V/$COH"
  echo "[coh-cv] $V $COH rc=$? date=$(date)"
done

echo "[coh-cv] variant=$V DONE date=$(date)"
