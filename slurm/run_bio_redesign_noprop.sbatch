#!/usr/bin/env bash
#SBATCH --job-name=smart-bio-noprop
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --array=0-1
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/bio-noprop-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/bio-noprop-%A_%a.out
# FIX-A-OFF ablation (BIO_ROUTER_REDESIGN.txt): graph PROPAGATION disabled, keeping
# only the router-side biology (Fix B gate + Fix C precision/PPR prior + Fix D
# learnable beta + Fix E Laplacian). Isolates whether over-smoothing was the cause
# of coexpr < random. Falsification test again: coexpr vs random vs none, 3 seeds.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p logs
DATASETS=(pancreas tabula_muris)
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
echo "[bio-noprop] start $(date) dataset=$DATASET"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.bio_redesign \
    --dataset "$DATASET" --modes none coexpr random --seeds 0 1 2 \
    --K 4 --epochs 60 --device cuda --prop off \
    --out "$PROJ/results_bio_redesign_noprop"
echo "[bio-noprop] DONE rc=$? $(date)"
