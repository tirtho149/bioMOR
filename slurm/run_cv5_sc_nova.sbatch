#!/usr/bin/env bash
#SBATCH --job-name=cv5-sc
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --array=0-63%64
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/cv5-sc-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/cv5-sc-%A_%a.out
# 5-FOLD CV, seed 42, 20% test / 10%-of-train val, epochs=100 patience=15.
# Single-cell GENERAL rows (Vanilla / Recursive / MoR). 8 variants x 8 datasets = 64 tasks.
# variant = idx/8 ; dataset = idx%8.  Output: results/cv5/sc/<variant>/<ds>.json
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

VARIANTS=(independent fixed_k2 fixed_k3 fixed expert_k2 expert_k3 shared token)
declare -a FLAGS
FLAGS[0]="--recursion_mode expert --no_share_weights"
FLAGS[1]="--recursion_mode fixed  --recursion_depth 2"
FLAGS[2]="--recursion_mode fixed  --recursion_depth 3"
FLAGS[3]="--recursion_mode fixed"
FLAGS[4]="--recursion_mode expert --recursion_depth 2"
FLAGS[5]="--recursion_mode expert --recursion_depth 3"
FLAGS[6]="--recursion_mode expert"
FLAGS[7]="--recursion_mode token"
DATASETS=(baron lung muraro oesophagus segerstolpe spleen tcell xin)

IDX=$SLURM_ARRAY_TASK_ID
V=${VARIANTS[$(( IDX / 8 ))]}; F="${FLAGS[$(( IDX / 8 ))]}"
DS=${DATASETS[$(( IDX % 8 ))]}
echo "[cv5-sc] variant=$V ds=$DS flags='$F' host=$(hostname) $(date)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.singlecell \
    --data data/singlecell --out "results/cv5/sc/$V" \
    --datasets "$DS" --cv_folds 5 --seed 42 \
    --epochs 100 --d_model 96 --n_markers 128 --batch_size 128 \
    --lr 0.001 --weight_decay 0.00001 --patience 15 --device cuda $F
echo "[cv5-sc] DONE rc=$? variant=$V ds=$DS $(date)"
