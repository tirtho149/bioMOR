#!/usr/bin/env bash
#SBATCH --job-name=smart-arch-bs
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --array=0-3
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/sc-arch-bs-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/sc-arch-bs-%A_%a.out
# Complete Table 1: the four architecture configs for Baron + Segerstolpe (3 seeds),
# which previously had only partial seeds. Output -> results_singlecell_arch/<v>/s<seed>/.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
NAMES=(shared fixed depth1 independent)
declare -a FLAGS
FLAGS[0]="--recursion_mode expert"
FLAGS[1]="--recursion_mode fixed"
FLAGS[2]="--recursion_mode expert --recursion_depth 1"
FLAGS[3]="--recursion_mode expert --no_share_weights"
V=${NAMES[$SLURM_ARRAY_TASK_ID]}; F="${FLAGS[$SLURM_ARRAY_TASK_ID]}"
echo "[arch-bs] variant=$V $(date)"; nvidia-smi -L || true
for SEED in 0 1 2; do
  $PY -m recursive_marker_transformer.singlecell \
      --data data/singlecell --out "results_singlecell_arch/$V/s$SEED" \
      --datasets baron segerstolpe \
      --epochs 150 --d_model 96 --n_markers 128 --batch_size 128 \
      --lr 0.001 --weight_decay 0.00001 --patience 15 --device cuda --seed $SEED $F
done
echo "[arch-bs] variant=$V DONE rc=$? $(date)"
