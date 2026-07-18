#!/usr/bin/env bash
#SBATCH --job-name=mo-provgraph
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --array=0-15%8
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/mo-provgraph-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/mo-provgraph-%A_%a.out
# Multi-omics bio-router using the PROVIDED Reactome pathway->pathway graph
# (data/<coh>/adjacency_matrix.csv) via --pathway_attn_bias, NOT a co-expression
# graph built on the sparse mut/CNV data. 2x2 factorial x 4 P-NET cohorts:
#   pool in {mean, sum(mutation-burden)}  x  attn in {off, on(provided CSV)}
# Same 5-fold CV (seed 42). idx = cohort*4 + config.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; source /work/mech-ai-scratch/tirtho/.venv/bin/activate 2>/dev/null || true
export PYTHONUNBUFFERED=1; export PYTHONPATH="$PROJ"; export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

COHS=(prostate blca stad brca)
i=$SLURM_ARRAY_TASK_ID
coh=${COHS[$(( i / 4 ))]}
cfg=$(( i % 4 ))
# brca is mutation-only; the other P-NET cohorts are mutation+CNV
CH=mut_cnv; [ "$coh" = brca ] && CH=mut

POOL=mean; ATTN=""
case $cfg in
  0) POOL=mean; ATTN="" ;;                      # baseline (current)
  1) POOL=sum;  ATTN="" ;;                      # +mutation-burden pooling
  2) POOL=mean; ATTN="--pathway_attn_bias" ;;   # +provided pathway graph
  3) POOL=sum;  ATTN="--pathway_attn_bias" ;;   # +both (the fix)
esac

echo "[provgraph] idx=$i coh=$coh ch=$CH pool=$POOL attn='${ATTN:-off}' host=$(hostname) $(date)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.pathway_tasks \
    --task "$coh" --channels "$CH" --marker_mode pathway --gene_interaction reactome \
    --recursion_mode expert --pathway_pool "$POOL" $ATTN \
    --n_markers 256 --d_model 128 --epochs 100 --lr 3e-4 --patience 15 \
    --batch_size 32 --cv_folds 5 --device cuda --out "results/cv5/mo_provided_graph"
echo "[provgraph] DONE idx=$i rc=$? $(date)"
