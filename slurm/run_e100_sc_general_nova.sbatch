#!/usr/bin/env bash
#SBATCH --job-name=e100-sc-gen
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00

#SBATCH --array=0-7%8
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/e100-sc-gen-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/e100-sc-gen-%A_%a.out
# 10-seed expansion of the single-cell GENERAL rows (Vanilla/Recursive/MoR-general/Token).
# arch13 already has seeds 0-2 (100/15); this adds seeds 3-9. 8 variants, one array task each.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
NAMES=(independent fixed_k2 fixed_k3 fixed expert_k2 expert_k3 shared token)
declare -a FLAGS
FLAGS[0]="--recursion_mode expert --no_share_weights"
FLAGS[1]="--recursion_mode fixed  --recursion_depth 2"
FLAGS[2]="--recursion_mode fixed  --recursion_depth 3"
FLAGS[3]="--recursion_mode fixed"
FLAGS[4]="--recursion_mode expert --recursion_depth 2"
FLAGS[5]="--recursion_mode expert --recursion_depth 3"
FLAGS[6]="--recursion_mode expert"
FLAGS[7]="--recursion_mode token"
V=${NAMES[$SLURM_ARRAY_TASK_ID]}; F="${FLAGS[$SLURM_ARRAY_TASK_ID]}"
echo "[e100-sc-gen] variant=$V flags='$F' host=$(hostname) $(date)"; nvidia-smi -L || true
for SEED in 3 4 5 6 7 8 9; do
  $PY -m recursive_marker_transformer.singlecell \
      --data data/singlecell --out "results_arch13/$V/s$SEED" \
      --datasets baron lung muraro oesophagus segerstolpe spleen tcell wang xin \
      --epochs 100 --d_model 96 --n_markers 128 --batch_size 128 \
      --lr 0.001 --weight_decay 0.00001 --patience 15 --device cuda \
      --seed $SEED $F
done
echo "[e100-sc-gen] variant=$V DONE rc=$? $(date)"
