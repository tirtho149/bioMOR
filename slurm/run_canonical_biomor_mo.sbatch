#!/usr/bin/env bash
#SBATCH --job-name=canon-biomor-mo
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --array=0-3%4
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/canon-biomor-mo-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/canon-biomor-mo-%A_%a.out
# CANONICAL bioMoR on MULTI-OMICS = "both sites", done PROPERLY with the PROVIDED
# Reactome graph (data/<coh>/adjacency_matrix.csv), NEVER a co-expression graph:
#   EMBEDDING site : --pathway_learned_graph --pathway_learned_fuse  (warm-start + prop
#                    + fuse a learned pathway graph from adjacency_matrix.csv)
#   ROUTER site    : --gene_interaction reactome  (per-pathway centrality prior, from the CSV)
#   + --pathway_attn_bias (pathway->pathway attention from the CSV)
#   + --pathway_pool sum  (mutation-burden pooling for the 0.4%-sparse mutation channel)
# Tokens are Reactome pathways (filtered_pathways.csv). CV5 seed 42, K=4.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; source /work/mech-ai-scratch/tirtho/.venv/bin/activate 2>/dev/null || true
export PYTHONUNBUFFERED=1; export PYTHONPATH="$PROJ"; export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
COHS=(prostate blca stad brca)
coh=${COHS[$SLURM_ARRAY_TASK_ID]}
CH=mut_cnv; [ "$coh" = brca ] && CH=mut
echo "[canon-mo] cohort=$coh ch=$CH host=$(hostname) $(date)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.pathway_tasks \
    --task "$coh" --channels "$CH" --marker_mode pathway --gene_interaction reactome \
    --pathway_learned_graph --pathway_learned_fuse --pathway_attn_bias \
    --pathway_pool sum --recursion_mode token \
    --n_markers 256 --d_model 128 --epochs 100 --patience 15 \
    --batch_size 32 --cv_folds 5 --device cuda --out "results/cv5/biomor_canonical_mo"
echo "[canon-mo] DONE cohort=$coh rc=$? $(date)"
