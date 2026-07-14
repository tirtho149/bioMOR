#!/usr/bin/env bash
#SBATCH --job-name=scale-sc-gen
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00

#SBATCH --array=0-9%10
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/scale-sc-gen-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/scale-sc-gen-%A_%a.out
# Scaling study, single-cell Vanilla + MoR-general at widths {96,136,192,272,352}, 100/15, 3 seeds.
# 2 models x 5 widths = 10 tasks. model=idx/5 (0 vanilla, 1 morgen); w=WIDTHS[idx%5].
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
WIDTHS=(96 136 192 272 352)
IDX=$SLURM_ARRAY_TASK_ID
W=${WIDTHS[$(( IDX % 5 ))]}
if [ $(( IDX / 5 )) -eq 0 ]; then MODEL=vanilla; SHARE="--no_share_weights"; else MODEL=morgen; SHARE="--share_weights"; fi
echo "[scale-sc-gen] model=$MODEL d_model=$W $(date) host=$(hostname)"; nvidia-smi -L || true
for SEED in 0 1 2; do
  $PY -m recursive_marker_transformer.singlecell \
      --data data/singlecell --out "results_scaling_sc/${MODEL}_d${W}/s$SEED" \
      --datasets baron lung muraro oesophagus segerstolpe spleen tcell wang xin \
      --epochs 100 --d_model "$W" --n_markers 128 --batch_size 128 \
      --lr 0.001 --weight_decay 0.00001 --patience 15 --device cuda \
      --recursion_mode expert --recursion_depth 4 --seed $SEED $SHARE
done
echo "[scale-sc-gen] DONE rc=$? $(date) model=$MODEL d=$W"
