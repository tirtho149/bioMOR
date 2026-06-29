#!/usr/bin/env bash
# ============================================================================
# MoR Tables 1 / 5 / 8: parameter-sharing schemes for the recursive stack, on the
# genomap datasets (K=6 recursions so middle_* schemes are distinct). One array
# task per scheme; reports acc + transformer params (param reduction vs depth).
#   0 shared(cycle,nu=1)  1 independent(cycle,nu=6)  2 cycle,nu=3
#   3 sequence,nu=3       4 middle_cycle,nu=3        5 middle_sequence,nu=3
#   out: results_sharing/<tag>/<dataset>.json
#   sbatch run_sharing.sbatch          # array 0-5
# ============================================================================
#SBATCH --job-name=smart-sharing
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --requeue
#SBATCH --array=0-5
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/sharing-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/sharing-%A_%a.out
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

TAGS=(shared_nu1 independent_nu6 cycle_nu3 sequence_nu3 middle_cycle_nu3 middle_sequence_nu3)
declare -a FL
FL[0]="--share_strategy cycle --n_unique_blocks 1"
FL[1]="--share_strategy cycle --n_unique_blocks 6"
FL[2]="--share_strategy cycle --n_unique_blocks 3"
FL[3]="--share_strategy sequence --n_unique_blocks 3"
FL[4]="--share_strategy middle_cycle --n_unique_blocks 3"
FL[5]="--share_strategy middle_sequence --n_unique_blocks 3"
T=${TAGS[$SLURM_ARRAY_TASK_ID]}; F="${FL[$SLURM_ARRAY_TASK_ID]}"
echo "[sharing] $T flags='$F' host=$(hostname) date=$(date)"; nvidia-smi -L || true

$PY -m recursive_marker_transformer.singlecell \
    --data data/singlecell --out "results_sharing/$T" \
    --datasets tabula_muris pancreas common_class prototype baron segerstolpe lung oesophagus spleen tcell \
    --epochs 80 --d_model 96 --n_markers 128 --batch_size 128 \
    --lr 0.001 --weight_decay 0.00001 --patience 12 --device cuda \
    --recursion_mode fixed --recursion_depth 6 --seed 0 $F
echo "[sharing] $T DONE rc=$? date=$(date)"
