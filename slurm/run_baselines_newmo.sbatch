#!/usr/bin/env bash
#SBATCH --job-name=base-newmo
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --array=0-3
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/base-newmo-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/base-newmo-%A_%a.out
# Reproduce Linear/RF/NearestCentroid baselines (5-fold, same folds as bioMoR) for the
# new multi-omics datasets -> results/cv5/baselines/<ds>/<method>_cv.json. CPU-only.
set -uo pipefail
cd /work/mech-ai-scratch/tirtho/RecusrsiveQFormer
source /work/mech-ai-scratch/tirtho/.venv/bin/activate 2>/dev/null || true
export PYTHONPATH=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
DS=(pan_meta_pri panmeta_response pan_meta_pri_3modal brca)
D=${DS[$SLURM_ARRAY_TASK_ID]}
echo "[base-newmo] ds=$D $(date) host=$(hostname)"
python -m recursive_marker_transformer.baselines11 --datasets "$D" --cv_folds 5 --out results/cv5/baselines
echo "[base-newmo] DONE rc=$? ds=$D $(date)"
