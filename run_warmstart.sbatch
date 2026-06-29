#!/usr/bin/env bash
# MoR Table 9 analogue (warm-start / uptraining): fixed-depth source -> warm-start a
# MoR model from its shared block vs train MoR from scratch. results_warmstart/<ds>.json
#   sbatch run_warmstart.sbatch
#SBATCH --job-name=smart-warmstart
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=14:00:00
#SBATCH --requeue
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/warmstart-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/warmstart-%j.out
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "[warmstart] host=$(hostname) date=$(date)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.warmstart \
    --data data/singlecell --out results_warmstart \
    --datasets tabula_muris pancreas common_class prototype baron segerstolpe lung oesophagus spleen tcell \
    --epochs 80 --d_model 96 --device cuda --seed 0
echo "[warmstart] DONE rc=$? date=$(date)"
