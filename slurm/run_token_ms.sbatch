#!/usr/bin/env bash
#SBATCH --job-name=smart-token-ms
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --array=0-4
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/token-ms-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/token-ms-%A_%a.out
# =============================================================================
# Multi-seed token-reduction (M-sweep) so the dedicated token table is multi-seed.
# One array task per M; each runs 3 seeds over the 10 genomap datasets and the 4
# P-NET cohorts. Output: results_msweep_ms/M<M>/s<seed>/<ds>.json (single-cell) and
# results_pathway_msweep_ms/M<M>/s<seed>/<tag>.json (cohorts).
#   sbatch run_token_ms.sbatch
# =============================================================================
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
M=(16 32 64 128 256); MM=${M[$SLURM_ARRAY_TASK_ID]}
echo "[token-ms] M=$MM host=$(hostname) date=$(date)"; nvidia-smi -L || true

COHORTS=("prostate mut_cnv" "blca mut_cnv" "stad mut_cnv" "panmeta_subtype expr")
for SEED in 0 1 2; do
  # single-cell suite (one call covers all 10 datasets)
  $PY -m recursive_marker_transformer.singlecell \
      --out "results_msweep_ms/M$MM/s$SEED" \
      --datasets tabula_muris pancreas common_class prototype baron segerstolpe lung oesophagus spleen tcell \
      --epochs 80 --d_model 96 --n_markers $MM --batch_size 128 --lr 0.001 \
      --weight_decay 0.00001 --patience 12 --device cuda --recursion_mode expert --seed $SEED
  # P-NET cohorts (learned-token router, matching the consolidated msweep)
  for CH in "${COHORTS[@]}"; do
    set -- $CH; TASK=$1; CHAN=$2
    BS=32; [ "$CHAN" = "expr" ] && BS=128
    $PY -m recursive_marker_transformer.pathway_tasks \
        --task $TASK --channels $CHAN --marker_mode router --gene_interaction none \
        --n_markers $MM --d_model 128 --epochs 40 --lr 3e-4 --patience 8 \
        --batch_size $BS --seed $SEED --device cuda \
        --out "results_pathway_msweep_ms/M$MM/s$SEED"
  done
done
echo "[token-ms] M=$MM DONE rc=$? date=$(date)"
