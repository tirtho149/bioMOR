#!/usr/bin/env bash
# ============================================================================
# SMART: Selective Marker-guided Adaptive Recursive Transformer
#        for Transcriptomic Classification
#
# Authors:
#   Koushik Howlader   - Iowa State University
#   Tirtho Roy         - Iowa State University
#   Md Tauhidul Islam  - Stanford University
#   Wei Le             - Iowa State University
#
# Copyright (c) 2026 The SMART Authors. All Rights Reserved.
#
# PROPRIETARY AND CONFIDENTIAL. Unauthorized use, copying, modification, or
# distribution of this file, in whole or in part, without the express written
# permission of the authors is STRICTLY PROHIBITED and will be prosecuted to
# the fullest extent permitted by law. See the LICENSE file for full terms.
# ============================================================================
#SBATCH --job-name=smart-tokenred
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --mail-user=tirtho@iastate.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/tokenred-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/tokenred-%j.out
# Token Reduction Validation: train the headline cohort model (config taken from
# results/main.json) and emit the structured validation artifact
#   results/token_reduction.json  (+ token_reduction_ranking.csv)
# then regenerate the paper .tex so the Token Reduction Validation tables fill.
#   sbatch run_token_reduction.sbatch
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "[tokenred] host=$(hostname) date=$(date)"; nvidia-smi -L
$PY -m recursive_marker_transformer.token_reduction \
    --config results/main.json --out results --top 50 --device cuda
$PY -m recursive_marker_transformer.make_paper --results results --outdir paper
echo "[tokenred] DONE rc=$? date=$(date)"
