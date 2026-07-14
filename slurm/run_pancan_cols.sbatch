#!/usr/bin/env bash
#SBATCH --job-name=pancan-cols
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --array=0-11%12
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/pancan-cols-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/pancan-cols-%A_%a.out
# Two NEW pan-cancer multi-omics columns for EVERY table row: pan_meta_pri (mut+cnv)
# and pancancer_meta_pri/panmeta_response (expression). One array task per variant,
# each looping both new cohorts x 3 seeds. Writes into results_pw13/<variant>/s<seed>/
# so build_ladder_table.py auto-picks the new-cohort tags for the two new columns.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

NAMES=(shared independent token fixed depth1 none fixed_k2 fixed_k3 expert_k2 expert_k3 fixed_cache expert_mcyc)
declare -a FLAGS
FLAGS[0]="--gene_interaction reactome --recursion_mode expert"
FLAGS[1]="--gene_interaction reactome --recursion_mode expert --no_share_weights"
FLAGS[2]="--gene_interaction reactome --recursion_mode token"
FLAGS[3]="--gene_interaction reactome --recursion_mode fixed"
FLAGS[4]="--gene_interaction reactome --recursion_mode expert --recursion_depth 1"
FLAGS[5]="--gene_interaction none --recursion_mode expert"
FLAGS[6]="--gene_interaction reactome --recursion_mode fixed  --recursion_depth 2"
FLAGS[7]="--gene_interaction reactome --recursion_mode fixed  --recursion_depth 3"
FLAGS[8]="--gene_interaction reactome --recursion_mode expert --recursion_depth 2"
FLAGS[9]="--gene_interaction reactome --recursion_mode expert --recursion_depth 3"
FLAGS[10]="--gene_interaction reactome --recursion_mode fixed  --recursion_depth 4 --step_cache"
FLAGS[11]="--gene_interaction reactome --recursion_mode expert --recursion_depth 4 --share_strategy middle_cycle"

V=${NAMES[$SLURM_ARRAY_TASK_ID]}
F="${FLAGS[$SLURM_ARRAY_TASK_ID]}"
echo "[pancan-cols] variant=$V flags='$F' host=$(hostname) date=$(date)"; nvidia-smi -L || true

for SEED in 0 1 2; do
  # pan_meta_pri: mutation + CNV multi-omics
  $PY -m recursive_marker_transformer.pathway_tasks \
      --task pan_meta_pri --channels mut_cnv --marker_mode pathway \
      --n_markers 256 --d_model 128 --epochs 40 --lr 3e-4 --patience 8 \
      --batch_size 32 --seed $SEED --device cuda \
      --out results_pw13/$V/s$SEED $F
  # pancancer_meta_pri: expression (panmeta_response = primary vs metastatic)
  $PY -m recursive_marker_transformer.pathway_tasks \
      --task panmeta_response --channels expr --marker_mode pathway \
      --n_markers 256 --d_model 128 --epochs 40 --lr 3e-4 --patience 8 \
      --batch_size 32 --seed $SEED --device cuda \
      --out results_pw13/$V/s$SEED $F
done
echo "[pancan-cols] variant=$V DONE rc=$? date=$(date)"
