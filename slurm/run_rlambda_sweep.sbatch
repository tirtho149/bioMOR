#!/usr/bin/env bash
#SBATCH --job-name=cv5-rlambda
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --array=0-29%12
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/cv5-rlambda-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/cv5-rlambda-%A_%a.out
# Sensitivity sweep for reviewer Q7: low-rank r of the learned graph and the
# smoothing trust-weight lambda. Indices 0-14 = 5 ranks x 3 datasets;
# 15-29 = 5 lambda x 3 datasets. r default=16, lambda default=0.2.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

DATA=(Baron Lung Tcell)
RANK=(4 8 16 32 64)
LAM=(0.05 0.1 0.2 0.4 0.8)
COMMON="--modes learned --cv_folds 5 --K 4 --epochs 100 --patience 15 --n_markers 128 --device cuda"

i=$SLURM_ARRAY_TASK_ID
if [ "$i" -lt 15 ]; then
  R=${RANK[$((i/3))]}; DS=${DATA[$((i%3))]}
  OUT="$PROJ/results/cv5/sensitivity/rank_r${R}"
  echo "[rlambda] rank sweep r=$R ds=$DS"
  "$PY" -m recursive_marker_transformer.bio_learned_genomap --dataset "$DS" \
        --learned_rank "$R" $COMMON --out "$OUT"
else
  j=$((i-15)); L=${LAM[$((j/3))]}; DS=${DATA[$((j%3))]}
  OUT="$PROJ/results/cv5/sensitivity/lambda_l${L}"
  echo "[rlambda] lambda sweep l=$L ds=$DS"
  "$PY" -m recursive_marker_transformer.bio_learned_genomap --dataset "$DS" \
        --prop_lambda_init "$L" $COMMON --out "$OUT"
fi
echo "[rlambda] done idx=$i rc=$?"
