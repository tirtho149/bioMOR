#!/usr/bin/env bash
#SBATCH --job-name=cv5-scale-sc-bio
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --array=0-39%40
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/cv5-scale-sc-bio-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/cv5-scale-sc-bio-%A_%a.out
# 5-FOLD CV model-SIZE sweep (seed 42, 100/15). Single-cell bioMoR (learned graph, K=4).
# 5 widths x 8 datasets = 40. w=idx/8 ; ds=idx%8.
# Output: results/cv5/scaling_sc_biomor/d<w>/<Dataset>/learned_cv.json
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
WIDTHS=(96 136 192 272 352)
DATASETS=(Baron Lung Muraro Oesophagus Segerstolpe Spleen Tcell Xin)
IDX=$SLURM_ARRAY_TASK_ID
W=${WIDTHS[$(( IDX / 8 ))]}; DS=${DATASETS[$(( IDX % 8 ))]}
echo "[cv5-scale-sc-bio] ds=$DS d=$W $(date) host=$(hostname)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.bio_learned_genomap \
    --dataset "$DS" --modes learned --cv_folds 5 --K 4 \
    --epochs 100 --patience 15 --d_model "$W" --n_markers 128 --device cuda \
    --out "$PROJ/results/cv5/scaling_sc_biomor/d${W}"
echo "[cv5-scale-sc-bio] DONE rc=$? $(date) ds=$DS d=$W"
