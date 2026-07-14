#!/usr/bin/env bash
#SBATCH --job-name=pwarch13-fill
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --array=0-5
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/pwarch13fill-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/pwarch13fill-%A_%a.out
# =============================================================================
# FILL the missing rungs (multi-omics / P-NET side): depth sweep N_R in {2,3}
# for Recursive(fixed) and general MoR(expert), plus KV-cache and middle-cycle
# share at K=4.  Writes into results_pw13/<variant>/s<seed>/<tag>.json.
# =============================================================================
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

NAMES=(fixed_k2 fixed_k3 expert_k2 expert_k3 fixed_cache expert_mcyc)
declare -a FLAGS
FLAGS[0]="--gene_interaction reactome --recursion_mode fixed  --recursion_depth 2"
FLAGS[1]="--gene_interaction reactome --recursion_mode fixed  --recursion_depth 3"
FLAGS[2]="--gene_interaction reactome --recursion_mode expert --recursion_depth 2"
FLAGS[3]="--gene_interaction reactome --recursion_mode expert --recursion_depth 3"
FLAGS[4]="--gene_interaction reactome --recursion_mode fixed  --recursion_depth 4 --step_cache"   # KV-cache needs fixed/token
FLAGS[5]="--gene_interaction reactome --recursion_mode expert --recursion_depth 4 --share_strategy middle_cycle"

V=${NAMES[$SLURM_ARRAY_TASK_ID]}
F="${FLAGS[$SLURM_ARRAY_TASK_ID]}"
echo "[pwarch13-fill] variant=$V flags='$F' host=$(hostname) date=$(date)"; nvidia-smi -L || true

COHORTS=(prostate blca stad brca)
for SEED in 0 1 2; do
  for TASK in "${COHORTS[@]}"; do
    $PY -m recursive_marker_transformer.pathway_tasks \
        --task $TASK --channels mut_cnv --marker_mode pathway \
        --n_markers 256 --d_model 128 --epochs 40 --lr 3e-4 --patience 8 \
        --batch_size 32 --seed $SEED --device cuda \
        --out results_pw13/$V/s$SEED $F
  done
done
echo "[pwarch13-fill] variant=$V DONE rc=$? date=$(date)"
