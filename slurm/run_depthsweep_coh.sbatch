#!/usr/bin/env bash
#SBATCH --job-name=smart-depthcoh
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=18:00:00
#SBATCH --requeue
#SBATCH --array=0-3
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/depthcoh-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/depthcoh-%A_%a.out
# Depth sweep (K=1..100) on the P-NET cohorts, so Table 2 reports every dataset.
# One array task per cohort; 2 seeds x the depth grid. results_depthsweep/<cohort>/.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
COH=(prostate blca stad panmeta_subtype)
D=${COH[$SLURM_ARRAY_TASK_ID]}
echo "[depthcoh] cohort=$D $(date)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.depth_sweep --datasets $D --seeds 0 1 --epochs 40 --device cuda
echo "[depthcoh] cohort=$D DONE rc=$? $(date)"
