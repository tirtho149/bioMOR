#!/usr/bin/env bash
#SBATCH --job-name=smart-arch-new4
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --array=0-6
#SBATCH --mail-user=tirtho@iastate.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/sc-arch-new4-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/sc-arch-new4-%A_%a.out
# =============================================================================
# Same architecture + marker-selection ablation as run_sc_arch.sbatch, but for the
# four Tier-2 genomap datasets added to the suite (lung, oesophagus, spleen, tcell),
# so they appear in every main-paper table with the other six. 3 seeds for mean+/-std.
# Output: results_singlecell_arch/<variant>/s<seed>/<dataset>.json
#   sbatch run_sc_arch_new4.sbatch
# =============================================================================
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

NAMES=(shared independent token fixed depth1 marker_random marker_var)
declare -a FLAGS
FLAGS[0]="--recursion_mode expert"
FLAGS[1]="--recursion_mode expert --no_share_weights"
FLAGS[2]="--recursion_mode token"
FLAGS[3]="--recursion_mode fixed"
FLAGS[4]="--recursion_mode expert --recursion_depth 1"
FLAGS[5]="--recursion_mode expert --marker_mode random"
FLAGS[6]="--recursion_mode expert --marker_mode variance"

V=${NAMES[$SLURM_ARRAY_TASK_ID]}
F="${FLAGS[$SLURM_ARRAY_TASK_ID]}"
echo "[sc-arch-new4] variant=$V flags='$F' host=$(hostname) date=$(date)"
nvidia-smi -L || true

for SEED in 0 1 2; do
  $PY -m recursive_marker_transformer.singlecell \
      --data data/singlecell --out "results_singlecell_arch/$V/s$SEED" \
      --datasets lung oesophagus spleen tcell \
      --epochs 150 --d_model 96 --n_markers 128 --batch_size 128 \
      --lr 0.001 --weight_decay 0.00001 --patience 15 --device cuda \
      --seed $SEED $F
done
echo "[sc-arch-new4] variant=$V DONE rc=$? date=$(date)"
