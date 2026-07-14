#!/usr/bin/env bash
#SBATCH --job-name=e100-pw
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --array=0-7%8
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/e100-pw-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/e100-pw-%A_%a.out
# APPLE-TO-APPLE re-run: multi-omics general ladder + two pan-cancer columns at epochs=100 patience=15.
# 8 variants (one array task each), looping cohorts {prostate,blca,stad (mut_cnv), pan_meta_pri (mut_cnv),
# panmeta_response (expr)} x seeds {0,1,2}. Writes results_pw13_e100/<variant>/s<seed>/<tag>.json.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
NAMES=(independent fixed_k2 fixed_k3 fixed expert_k2 expert_k3 shared token)
declare -a FLAGS
FLAGS[0]="--gene_interaction reactome --recursion_mode expert --no_share_weights"
FLAGS[1]="--gene_interaction reactome --recursion_mode fixed  --recursion_depth 2"
FLAGS[2]="--gene_interaction reactome --recursion_mode fixed  --recursion_depth 3"
FLAGS[3]="--gene_interaction reactome --recursion_mode fixed"
FLAGS[4]="--gene_interaction reactome --recursion_mode expert --recursion_depth 2"
FLAGS[5]="--gene_interaction reactome --recursion_mode expert --recursion_depth 3"
FLAGS[6]="--gene_interaction reactome --recursion_mode expert"
FLAGS[7]="--gene_interaction reactome --recursion_mode token"
V=${NAMES[$SLURM_ARRAY_TASK_ID]}; F="${FLAGS[$SLURM_ARRAY_TASK_ID]}"
echo "[e100-pw] variant=$V flags='$F' host=$(hostname) $(date)"; nvidia-smi -L || true
for SEED in 0 1 2 3 4 5 6 7 8 9; do
  for T in prostate blca stad pan_meta_pri; do
    $PY -m recursive_marker_transformer.pathway_tasks \
        --task $T --channels mut_cnv --marker_mode pathway \
        --n_markers 256 --d_model 128 --epochs 100 --lr 3e-4 --patience 15 \
        --batch_size 32 --seed $SEED --device cuda --out results_pw13_e100/$V/s$SEED $F
  done
  $PY -m recursive_marker_transformer.pathway_tasks \
      --task panmeta_response --channels expr --marker_mode pathway \
      --n_markers 256 --d_model 128 --epochs 100 --lr 3e-4 --patience 15 \
      --batch_size 32 --seed $SEED --device cuda --out results_pw13_e100/$V/s$SEED $F
done
echo "[e100-pw] variant=$V DONE rc=$? $(date)"
