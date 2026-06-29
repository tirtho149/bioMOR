#!/usr/bin/env bash
# ============================================================================
# MoR Tables 10 / 11: expert-choice (T10) and token-choice (T11) routers under
# different routing configurations (router_type linear/MLP, temperature, alpha,
# balancing) on the genomap datasets. out: results_routing/<tag>/<dataset>.json
#   sbatch run_routing.sbatch          # array 0-5
# ============================================================================
#SBATCH --job-name=smart-routing
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=14:00:00
#SBATCH --requeue
#SBATCH --array=0-5
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/routing-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/routing-%A_%a.out
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

TAGS=(exp_linear exp_mlp exp_temp2 tok_linear tok_mlp tok_bal01)
declare -a FL
FL[0]="--recursion_mode expert --router_type linear"
FL[1]="--recursion_mode expert --router_type mlp"
FL[2]="--recursion_mode expert --router_type linear --router_temp 2.0"
FL[3]="--recursion_mode token --router_type linear"
FL[4]="--recursion_mode token --router_type mlp"
FL[5]="--recursion_mode token --router_type linear --router_balance_coeff 0.01"
T=${TAGS[$SLURM_ARRAY_TASK_ID]}; F="${FL[$SLURM_ARRAY_TASK_ID]}"
echo "[routing] $T flags='$F' host=$(hostname) date=$(date)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.singlecell \
    --data data/singlecell --out "results_routing/$T" \
    --datasets tabula_muris pancreas common_class prototype baron segerstolpe lung oesophagus spleen tcell \
    --epochs 80 --d_model 96 --n_markers 128 --batch_size 128 \
    --lr 0.001 --weight_decay 0.00001 --patience 12 --device cuda \
    --recursion_depth 4 --seed 0 $F
echo "[routing] $T DONE rc=$? date=$(date)"
