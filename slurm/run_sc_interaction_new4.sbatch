#!/usr/bin/env bash
#SBATCH --job-name=smart-bio-new4
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --array=0-11
#SBATCH --mail-user=tirtho@iastate.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/sc-bio-new4-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/sc-bio-new4-%A_%a.out
# =============================================================================
# Biology-informed-router ablation (none/coexpr/random x 3 seeds) for the four
# Tier-2 datasets (lung, oesophagus, spleen, tcell), matching run_sc_interaction.sbatch.
# One array task per (dataset, mode). Also produces the per-dataset metadata the
# main-paper overview/headline tables read.
# Output: results_sc_interaction/<dataset>__<mode>__seed<s>.json
#   sbatch run_sc_interaction_new4.sbatch
# =============================================================================
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

DATASETS=(lung lung lung oesophagus oesophagus oesophagus spleen spleen spleen tcell tcell tcell)
MODES=(none coexpr random none coexpr random none coexpr random none coexpr random)
DS=${DATASETS[$SLURM_ARRAY_TASK_ID]}
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}

echo "[sc-bio-new4] dataset=$DS mode=$MODE host=$(hostname) date=$(date)"
nvidia-smi -L || true

$PY -m recursive_marker_transformer.sc_interaction \
    --data data/singlecell --out results_sc_interaction \
    --datasets "$DS" --modes "$MODE" --seeds 0 1 2 \
    --epochs 150 --d_model 96 --n_markers 128 --batch_size 128 \
    --lr 0.001 --weight_decay 0.00001 --patience 15 \
    --beta 1.0 --knn 16 --anneal true --device cuda
echo "[sc-bio-new4] dataset=$DS mode=$MODE DONE rc=$? date=$(date)"
