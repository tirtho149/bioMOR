#!/usr/bin/env bash
#SBATCH --job-name=cv5-base
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --array=0-10%11
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/cv5-base-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/cv5-base-%A_%a.out
# 5-fold CV classical baselines (Linear / RandomForest / NearestCentroid), same folds
# as bioMoR. 11 datasets, one per array task. -> results/cv5/baselines/<ds>/<m>_cv.json
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
DS=(Baron Lung Muraro Oesophagus Segerstolpe Spleen Tcell Xin prostate blca stad)
D=${DS[$SLURM_ARRAY_TASK_ID]}
echo "[cv5-base] ds=$D $(date) host=$(hostname)"
$PY -m recursive_marker_transformer.baselines11 --datasets "$D" --cv_folds 5 --out results/cv5/baselines
echo "[cv5-base] DONE rc=$? ds=$D $(date)"
