#!/usr/bin/env bash
#SBATCH --job-name=canon-biomor-sc
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --array=0-7%8
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/canon-biomor-sc-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/canon-biomor-sc-%A_%a.out
# CANONICAL bioMoR = bio_both (interaction injected at BOTH sites: gene-embedding
# warm-start + router centrality prior). 8 single-cell table datasets, CV5 seed 42, K=4.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; source /work/mech-ai-scratch/tirtho/.venv/bin/activate 2>/dev/null || true
export PYTHONUNBUFFERED=1; export PYTHONPATH="$PROJ"; export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
DS=(Segerstolpe Lung Oesophagus Baron Muraro Tcell Spleen Xin)
D=${DS[$SLURM_ARRAY_TASK_ID]}
echo "[canon-sc] dataset=$D host=$(hostname) $(date)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.bio_learned_genomap --dataset "$D" \
    --modes bio_both --cv_folds 5 --K 4 --epochs 100 --patience 15 \
    --n_markers 128 --device cuda --out "results/cv5/biomor_canonical"
echo "[canon-sc] DONE dataset=$D rc=$? $(date)"
