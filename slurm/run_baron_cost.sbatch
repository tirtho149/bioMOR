#!/usr/bin/env bash
#SBATCH --job-name=baron-cost
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/baron-cost-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/baron-cost-%j.out
# Timed per-epoch training on Baron for Vanilla / Recursive / MoR / bioMoR (Table-2 configs)
# -> results/cv5/curves/baron_cost.json, for the compute-cost figures.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
echo "[baron-cost] $(date) host=$(hostname)"; nvidia-smi -L || true
"$PY" scripts/make_baron_cost.py
echo "[baron-cost] DONE rc=$? $(date)"
