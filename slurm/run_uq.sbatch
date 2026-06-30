#!/usr/bin/env bash
#SBATCH --job-name=smart-uq
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --array=0-5
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/uq-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/uq-%A_%a.out
# =============================================================================
# Log-prob / calibration UQ sweep -- one array task per configuration, 3 seeds,
# genomap single-cell suite + P-NET cohorts. Output: results_uq/<config>/s<seed>/.
#   sbatch run_uq.sbatch
# =============================================================================
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
CFG=(bio none adaptive fixed depth1 independent)
C=${CFG[$SLURM_ARRAY_TASK_ID]}
echo "[uq] config=$C host=$(hostname) date=$(date)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.uq_sweep --config $C --seeds 0 1 2 \
    --epochs 80 --cohorts --device cuda
echo "[uq] config=$C DONE rc=$? date=$(date)"
