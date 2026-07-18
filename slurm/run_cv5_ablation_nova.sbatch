#!/usr/bin/env bash
#SBATCH --job-name=cv5-ablation
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --array=0-158%48
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/cv5-abl-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/cv5-abl-%A_%a.out
# Six ablation families (1->6), SAME 5-fold CV protocol (seed 42, 100/15). Array index
# = one (condition x dataset); scripts/ablate_cv5.py dispatches to the right --cv_folds 5 CLI.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "[cv5-abl] idx=$SLURM_ARRAY_TASK_ID host=$(hostname) $(date)"; nvidia-smi -L || true
$PY scripts/ablate_cv5.py --index "$SLURM_ARRAY_TASK_ID"
echo "[cv5-abl] DONE rc=$? idx=$SLURM_ARRAY_TASK_ID $(date)"
