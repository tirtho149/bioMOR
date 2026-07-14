#!/usr/bin/env bash
#SBATCH --job-name=arch13-fill
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --array=0-5
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/arch13fill-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/arch13fill-%A_%a.out
# =============================================================================
# FILL the missing rungs of the MoR-format ladder (single-cell side):
#   depth sweep N_R in {2,3} for Recursive(fixed) and general MoR(expert),
#   plus KV-cache (step_cache) and middle-cycle share strategy at K=4.
# Writes into results_arch13/<variant>/s<seed>/<ds>.json (picked up by extractor).
# =============================================================================
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

NAMES=(fixed_k2 fixed_k3 expert_k2 expert_k3 fixed_cache expert_mcyc)
declare -a FLAGS
FLAGS[0]="--recursion_mode fixed  --recursion_depth 2"
FLAGS[1]="--recursion_mode fixed  --recursion_depth 3"
FLAGS[2]="--recursion_mode expert --recursion_depth 2"
FLAGS[3]="--recursion_mode expert --recursion_depth 3"
FLAGS[4]="--recursion_mode fixed  --recursion_depth 4 --step_cache"   # KV-cache needs fixed/token (not expert)
FLAGS[5]="--recursion_mode expert --recursion_depth 4 --share_strategy middle_cycle"

V=${NAMES[$SLURM_ARRAY_TASK_ID]}
F="${FLAGS[$SLURM_ARRAY_TASK_ID]}"
echo "[arch13-fill] variant=$V flags='$F' host=$(hostname) date=$(date)"; nvidia-smi -L || true

for SEED in 0 1 2; do
  $PY -m recursive_marker_transformer.singlecell \
      --data data/singlecell --out "results_arch13/$V/s$SEED" \
      --datasets baron lung muraro oesophagus segerstolpe spleen tcell wang xin \
      --epochs 100 --d_model 96 --n_markers 128 --batch_size 128 \
      --lr 0.001 --weight_decay 0.00001 --patience 15 --device cuda \
      --seed $SEED $F
done
echo "[arch13-fill] variant=$V DONE rc=$? date=$(date)"
