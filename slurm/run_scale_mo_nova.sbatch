#!/usr/bin/env bash
#SBATCH --job-name=scale-mo
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00

#SBATCH --array=0-24%16
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/scale-mo-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/scale-mo-%A_%a.out
# Scaling study, multi-omics (P-NET) at widths {96,136,192,272,352}, 100/15, 3 seeds.
# 3 models x 5 widths = 15 general tasks (idx 0-14) + 5 widths x ... bioMoR handled here too.
# idx 0-4  = vanilla (pathway, no-share) ; 5-9 = morgen (pathway, share) ; 10-14 = bioMoR (bio_redesign).
# Each task loops cohorts {prostate,blca,stad} x seeds {0,1,2}. w=WIDTHS[idx%5].
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
WIDTHS=(96 136 192 272 352)
COHORTS=(prostate blca stad)
IDX=$SLURM_ARRAY_TASK_ID
W=${WIDTHS[$(( IDX % 5 ))]}; GRP=$(( IDX / 5 ))
echo "[scale-mo] grp=$GRP d_model=$W $(date) host=$(hostname)"; nvidia-smi -L || true
for SEED in 0 1 2; do
  for COH in "${COHORTS[@]}"; do
    case $GRP in
      0) $PY -m recursive_marker_transformer.pathway_tasks --task $COH --channels mut_cnv \
            --marker_mode pathway --n_markers 256 --d_model "$W" --epochs 100 --lr 3e-4 --patience 15 \
            --batch_size 32 --seed $SEED --device cuda --out results_scaling_mo/vanilla_d${W}/s$SEED \
            --gene_interaction reactome --recursion_mode expert --no_share_weights ;;
      1) $PY -m recursive_marker_transformer.pathway_tasks --task $COH --channels mut_cnv \
            --marker_mode pathway --n_markers 256 --d_model "$W" --epochs 100 --lr 3e-4 --patience 15 \
            --batch_size 32 --seed $SEED --device cuda --out results_scaling_mo/morgen_d${W}/s$SEED \
            --gene_interaction reactome --recursion_mode expert ;;
      2) $PY -m recursive_marker_transformer.bio_redesign_curated --family pnet --cohort $COH \
            --task response --modes learned --seeds $SEED --K 4 --epochs 100 --patience 15 \
            --cap_genes 3000 --d_model "$W" --device cuda --out "$PROJ/results_scaling_mo_biomor/d${W}" ;;
    esac
  done
done
echo "[scale-mo] DONE rc=$? $(date) grp=$GRP d=$W"
