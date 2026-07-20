#!/usr/bin/env bash
#SBATCH --job-name=baron7-smoke
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --array=0-1%2
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/baron7-smoke-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/baron7-smoke-%A_%a.out
# Baron 7-COMBINATION biology ablation (parallel). Three binary mechanisms on a shared learned
# gene graph, all in token mode: E=embedding smoothing, R=router graph-conv, A=attention bias
# (the mechanism un-gated for single-cell). The 7 non-empty combinations of {E,R,A}.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
cd "$PROJ"
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

COMBOS=(A ERA)
M="${COMBOS[$SLURM_ARRAY_TASK_ID]}"
OUT="$PROJ/results/cv5/baron_7combo_smoke"

echo "[baron7-smoke] $(date) combo=$M host=$(hostname)"; nvidia-smi -L || true
python -m recursive_marker_transformer.bio_learned_genomap \
    --dataset Baron --modes "$M" --cv_folds 2 --K 4 --epochs 3 --patience 2 \
    --n_markers 128 --seed 42 --device cuda --out "$OUT"
echo "[baron7-smoke] DONE rc=$? combo=$M $(date)"
