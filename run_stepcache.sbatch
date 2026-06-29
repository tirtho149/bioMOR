#!/usr/bin/env bash
# ============================================================================
# MoR Tables 2 / 12 / 13: step-cache (reuse step-1 attention K/V across recursions
# vs recompute) on the genomap datasets, fixed-depth recursion. Array: 0=off, 1=on.
#   out: results_stepcache/<off|on>/<dataset>.json
#   sbatch run_stepcache.sbatch
# ============================================================================
#SBATCH --job-name=smart-stepcache
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --array=0-1
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/stepcache-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/stepcache-%A_%a.out
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

if [ "$SLURM_ARRAY_TASK_ID" = "1" ]; then TAG=on; CACHE="--step_cache"; else TAG=off; CACHE=""; fi
echo "[stepcache] $TAG host=$(hostname) date=$(date)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.singlecell \
    --data data/singlecell --out "results_stepcache/$TAG" \
    --datasets tabula_muris pancreas common_class prototype baron segerstolpe lung oesophagus spleen tcell \
    --epochs 80 --d_model 96 --n_markers 128 --batch_size 128 \
    --lr 0.001 --weight_decay 0.00001 --patience 12 --device cuda \
    --recursion_mode fixed --recursion_depth 4 --seed 0 $CACHE
echo "[stepcache] $TAG DONE rc=$? date=$(date)"
