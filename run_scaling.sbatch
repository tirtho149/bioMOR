#!/usr/bin/env bash
# ============================================================================
# MoR Table 3 / 6 / 7 + Figure 3: scaling of MoR vs Recursive vs Vanilla across
# model sizes on the genomap datasets. Each array task = one (arch x d_model) cell,
# run over all genomap datasets, 1 seed. Reports acc + param counts + mean depth.
#   arch:    vanilla   = K independent blocks, fixed depth (standard transformer)
#            recursive = 1 shared block x K, fixed depth (weight-shared recursion)
#            mor       = 1 shared block x K, expert-choice routing (SMART)
#   sizes:   d_model in {48, 96, 192, 384}
#   out:     results_scaling/<arch>_d<d>/<dataset>.json
#   sbatch run_scaling.sbatch          # array 0-11 (3 archs x 4 sizes)
# ============================================================================
#SBATCH --job-name=smart-scaling
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --requeue
#SBATCH --array=0-11
#SBATCH --mail-user=tirtho@iastate.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/scaling-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/scaling-%A_%a.out
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

ARCHS=(vanilla recursive mor)
declare -A AFLAG=(
 [vanilla]="--recursion_mode fixed --no_share_weights"
 [recursive]="--recursion_mode fixed"
 [mor]="--recursion_mode expert"
)
SIZES=(48 96 192 384)

A=$(( SLURM_ARRAY_TASK_ID / 4 ))     # 0..2 arch
S=$(( SLURM_ARRAY_TASK_ID % 4 ))     # 0..3 size
ARCH=${ARCHS[$A]}; D=${SIZES[$S]}; F="${AFLAG[$ARCH]}"
echo "[scaling] arch=$ARCH d_model=$D flags='$F' host=$(hostname) date=$(date)"
nvidia-smi -L || true

$PY -m recursive_marker_transformer.singlecell \
    --data data/singlecell --out "results_scaling/${ARCH}_d${D}" \
    --datasets tabula_muris pancreas common_class prototype baron segerstolpe lung oesophagus spleen tcell \
    --epochs 80 --d_model "$D" --n_markers 128 --batch_size 128 \
    --lr 0.001 --weight_decay 0.00001 --patience 12 --device cuda \
    --recursion_depth 4 --seed 0 $F
echo "[scaling] arch=$ARCH d=$D DONE rc=$? date=$(date)"
