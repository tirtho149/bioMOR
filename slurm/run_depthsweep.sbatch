#!/usr/bin/env bash
#SBATCH --job-name=smart-depthsweep
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=18:00:00
#SBATCH --requeue
#SBATCH --array=0-9
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/depthsweep-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/depthsweep-%A_%a.out
# =============================================================================
# Formal recursion-depth selection: sweep fixed depth K in {1..100} and pick K* by a
# held-out-validation one-standard-error rule (early stopping over depth). One array
# task per dataset; 2 seeds x 18 depths. results_depthsweep/<dataset>/K<k>_s<seed>.json
#   sbatch slurm/run_depthsweep.sbatch     # then: python -m ...depth_sweep --report
# =============================================================================
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
DS=(tabula_muris pancreas common_class prototype baron segerstolpe lung oesophagus spleen tcell)
D=${DS[$SLURM_ARRAY_TASK_ID]}
echo "[depthsweep] dataset=$D host=$(hostname) date=$(date)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.depth_sweep --datasets $D --seeds 0 1 --epochs 60 --device cuda
echo "[depthsweep] dataset=$D DONE rc=$? date=$(date)"
