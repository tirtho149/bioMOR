#!/usr/bin/env bash
#SBATCH --job-name=smart-sc-bio
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
#SBATCH --mail-user=tirtho@iastate.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/sc-bio-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/sc-bio-%A_%a.out
# =============================================================================
# HEADLINE EXPERIMENT: biology-informed-router ablation on the genomap single-cell
# datasets (Tabula Muris + pancreas ONLY). Three router-prior modes
#   none   -- original SMART router
#   coexpr -- genomap gene-gene co-expression-centrality prior (proposed)
#   random -- degree-matched random-graph control
# x 3 seeds, genomap-paper protocol (lr 1e-3, wd 1e-5, batch 128, 150 epochs).
# One array task per (dataset, mode); each runs all 3 seeds. Resumable.
# Output: results_sc_interaction/<dataset>__<mode>__seed<s>.json
#   sbatch run_sc_interaction.sbatch
# =============================================================================
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

DATASETS=(tabula_muris tabula_muris tabula_muris pancreas pancreas pancreas)
MODES=(none coexpr random none coexpr random)
DS=${DATASETS[$SLURM_ARRAY_TASK_ID]}
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}

echo "[sc-bio] dataset=$DS mode=$MODE host=$(hostname) date=$(date)"
nvidia-smi -L || true

$PY -m recursive_marker_transformer.sc_interaction \
    --data data/singlecell --out results_sc_interaction \
    --datasets "$DS" --modes "$MODE" --seeds 0 1 2 \
    --epochs 150 --d_model 96 --n_markers 128 --batch_size 128 \
    --lr 0.001 --weight_decay 0.00001 --patience 15 \
    --beta 1.0 --knn 16 --anneal true --device cuda
echo "[sc-bio] dataset=$DS mode=$MODE DONE rc=$? date=$(date)"
