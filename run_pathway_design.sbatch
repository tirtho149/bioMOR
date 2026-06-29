#!/usr/bin/env bash
# Run the remaining design-decision experiments on the GitHub pathway cohorts so they
# appear in every table (not just the arch ablation): sharing schemes, step-cache, scaling.
#   array 0=sharing  1=stepcache  2=scaling
#SBATCH --job-name=smart-pwdesign
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --requeue
#SBATCH --array=0-2
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/pwdesign-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/pwdesign-%A_%a.out
set -uo pipefail
cd /work/mech-ai-scratch/tirtho/RecusrsiveQFormer
export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
# cohort -> channels
declare -A CH=( [prostate]=mut_cnv [blca]=mut_cnv [stad]=mut_cnv [panmeta_subtype]=expr )
COH=(prostate blca stad panmeta_subtype)
COM="--marker_mode pathway --gene_interaction reactome --epochs 40 --d_model 128 \
     --n_markers 256 --lr 3e-4 --patience 8 --device cuda --seed 0"
run(){ local t=$1 out=$2; shift 2; local bs=64; [ "$t" = panmeta_subtype ] && bs=128
  $PY -m recursive_marker_transformer.pathway_tasks --task "$t" --channels "${CH[$t]}" \
      --out "$out" --batch_size $bs $COM "$@"; }

case $SLURM_ARRAY_TASK_ID in
 0) for t in "${COH[@]}"; do
      run $t results_pathway_sharing/shared_nu1     --recursion_mode fixed --recursion_depth 6 --share_strategy cycle --n_unique_blocks 1
      run $t results_pathway_sharing/independent_nu6 --recursion_mode fixed --recursion_depth 6 --share_strategy cycle --n_unique_blocks 6
      run $t results_pathway_sharing/cycle_nu3       --recursion_mode fixed --recursion_depth 6 --share_strategy cycle --n_unique_blocks 3
      run $t results_pathway_sharing/sequence_nu3    --recursion_mode fixed --recursion_depth 6 --share_strategy sequence --n_unique_blocks 3
      run $t results_pathway_sharing/middle_cycle_nu3 --recursion_mode fixed --recursion_depth 6 --share_strategy middle_cycle --n_unique_blocks 3
      run $t results_pathway_sharing/middle_sequence_nu3 --recursion_mode fixed --recursion_depth 6 --share_strategy middle_sequence --n_unique_blocks 3
    done ;;
 1) for t in "${COH[@]}"; do
      run $t results_pathway_stepcache/off --recursion_mode fixed
      run $t results_pathway_stepcache/on  --recursion_mode fixed --step_cache
    done ;;
 2) for t in prostate blca stad; do for D in 64 128 256; do
      run $t results_pathway_scaling/vanilla_d$D   --recursion_mode fixed --no_share_weights --d_model $D --recursion_depth 4
      run $t results_pathway_scaling/recursive_d$D --recursion_mode fixed                     --d_model $D --recursion_depth 4
      run $t results_pathway_scaling/mor_d$D       --recursion_mode expert                    --d_model $D --recursion_depth 4
    done; done ;;
esac
echo "[pwdesign] task=$SLURM_ARRAY_TASK_ID DONE rc=$? date=$(date)"
