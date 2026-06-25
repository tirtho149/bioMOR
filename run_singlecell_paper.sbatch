#!/usr/bin/env bash
#SBATCH --job-name=smart-sc-paper
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --array=0-2
#SBATCH --mail-user=tirtho@iastate.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/sc-paper-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/sc-paper-%A_%a.out
# =============================================================================
# Single-cell CELL-RECOGNITION exactly as in the genomap paper (Islam & Xing,
# Nat. Commun. 2023, s41467-023-36383-6): cell-type classification on the
# capsule datasets (genomaps: the paper's 5-10% HVG + entropy cartography,
# precomputed in the capsule) using the paper's exact protocol --
#   * data: precomputed genomap features (33x33=1089, 44x44=1936, ...)
#   * split: shipped where present (Tabula Muris 70/30, pancreas integration),
#            else stratified 70/30 ("70% train, rest test" -- paper)
#   * optimizer: Adam + weight decay, lr 0.001, weight_decay 1e-5, batch 128,
#                150 epochs, shuffle each epoch (paper Methods / Suppl. Table S2)
#   * metric: ACCURACY (paper headline)
# Run for all THREE model types. One array task per variant (0=shared,
# 1=independent, 2=mor).
# Output: results_singlecell_paper/<variant>/<dataset>.json
#   sbatch run_singlecell_paper.sbatch
# =============================================================================
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

VARIANTS=(shared independent mor)
V=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
case "$V" in
  shared)      MODEFLAGS="--recursion_mode expert" ;;
  independent) MODEFLAGS="--recursion_mode expert --no_share_weights" ;;
  mor)         MODEFLAGS="--recursion_mode token" ;;
esac

echo "[sc-paper] variant=$V host=$(hostname) date=$(date)"
nvidia-smi -L

# Paper hyperparameters: lr 0.001, weight_decay 1e-5, batch 128, 150 epochs.
$PY -m recursive_marker_transformer.singlecell \
    --data data/singlecell --out "results_singlecell_paper/$V" \
    --datasets tabula_muris common_class prototype pancreas \
    --epochs 150 --d_model 96 --n_markers 128 --batch_size 128 \
    --lr 0.001 --weight_decay 0.00001 --patience 15 --device cuda $MODEFLAGS
echo "[sc-paper] variant=$V DONE rc=$? date=$(date)"
