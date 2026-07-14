#!/usr/bin/env bash
#SBATCH --job-name=cv5-scale-mo-gen
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --array=0-29%30
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/cv5-scale-mo-gen-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/cv5-scale-mo-gen-%A_%a.out
# 5-FOLD CV model-SIZE sweep (seed 42, 100/15). Multi-omics Vanilla + MoR-general (P-NET cohorts).
# 2 models x 5 widths x 3 cohorts = 30. model=idx/15 ; w=(idx%15)/3 ; coh=(idx%15)%3.
# Output: results_cv5/scaling_mo/<model>_d<w>/<tag>_cv.json
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
WIDTHS=(96 136 192 272 352)
COHORTS=(prostate blca stad)
IDX=$SLURM_ARRAY_TASK_ID
if [ $(( IDX / 15 )) -eq 0 ]; then MODEL=vanilla; SHARE="--no_share_weights"; else MODEL=morgen; SHARE=""; fi
REM=$(( IDX % 15 )); W=${WIDTHS[$(( REM / 3 ))]}; COH=${COHORTS[$(( REM % 3 ))]}
echo "[cv5-scale-mo-gen] model=$MODEL d=$W coh=$COH $(date) host=$(hostname)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.pathway_tasks --task "$COH" --channels mut_cnv \
    --marker_mode pathway --n_markers 256 --d_model "$W" --epochs 100 --lr 3e-4 --patience 15 \
    --batch_size 32 --cv_folds 5 --device cuda --out "results_cv5/scaling_mo/${MODEL}_d${W}" \
    --gene_interaction reactome --recursion_mode expert $SHARE
echo "[cv5-scale-mo-gen] DONE rc=$? $(date) model=$MODEL d=$W coh=$COH"
