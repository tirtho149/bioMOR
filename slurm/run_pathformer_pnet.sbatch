#!/usr/bin/env bash
#SBATCH --job-name=smart-pathformer
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/pathformer-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/pathformer-%A_%a.out
# PATHFORMER-STYLE variant (Lu Lab, github.com/lulab/Pathformer) on the same P-NET
# cohorts: tokens are PATHWAYS (pool member genes), structured by a pathway CROSSTALK
# network (Pathformer Module 1). We use the cohort's shipped Reactome pathway<->pathway
# adjacency as the crosstalk graph (offline; not the 4-DB BinoX network). Compare
# crosstalk ON (pathway_attn_bias) vs OFF, both token-routing, 5 seeds per cohort.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
COHORTS=(prostate blca stad brca)
TASK="${COHORTS[$SLURM_ARRAY_TASK_ID]}"
echo "[pathformer] start $(date) task=$TASK"; nvidia-smi -L || true
for SEED in 0 1 2 3 4; do
  OUT="$PROJ/results_pathformer/seed${SEED}"
  # Pathformer-style: pathway tokens + crosstalk network (attn bias ON)
  $PY -m recursive_marker_transformer.pathway_tasks --task "$TASK" --channels mut \
      --marker_mode pathway --recursion_mode token --pathway_attn_bias \
      --gene_interaction none --seed "$SEED" --epochs 60 --device cuda --out "$OUT"
  # Ablation: same pathway tokens, NO crosstalk network
  $PY -m recursive_marker_transformer.pathway_tasks --task "$TASK" --channels mut \
      --marker_mode pathway --recursion_mode token \
      --gene_interaction none --seed "$SEED" --epochs 60 --device cuda --out "$OUT"
done
echo "[pathformer] DONE rc=$? $(date)"
