#!/usr/bin/env bash
#SBATCH --job-name=biomor-ladder-sc
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --array=0-26%10
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/biomor-ladder-sc-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/biomor-ladder-sc-%A_%a.out
# bioMoR recursion ladder on single-cell: learned gene-graph at K=1/2/3 (K=4 = headline, done).
# 9 datasets x 3 K = 27 tasks; idx -> K=1+idx/9, dataset=idx%9. mode=learned, seeds 0 1 2.
# Writes to results_biomor_ladder/k<K>/<dataset>/learned_s<seed>.json (NEVER touches K=4 headline).
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
DATASETS=(Lung Oesophagus Spleen Tcell Baron Muraro Segerstolpe Wang Xin)
IDX=$SLURM_ARRAY_TASK_ID
K=$(( 1 + IDX / 9 ))
DS="${DATASETS[$(( IDX % 9 ))]}"
echo "[biomor-ladder-sc] start $(date) dataset=$DS K=$K"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.bio_learned_genomap \
    --dataset "$DS" --modes learned --seeds 0 1 2 \
    --K "$K" --epochs 60 --device cuda \
    --out "$PROJ/results_biomor_ladder/k${K}"
echo "[biomor-ladder-sc] DONE rc=$? $(date) dataset=$DS K=$K"
