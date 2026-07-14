#!/usr/bin/env bash
#SBATCH --job-name=cv5-scale-mo-bio
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --array=0-14%15
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/cv5-scale-mo-bio-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/cv5-scale-mo-bio-%A_%a.out
# 5-FOLD CV model-SIZE sweep (seed 42, 100/15). Multi-omics bioMoR (P-NET, learned graph, K=4).
# 5 widths x 3 cohorts = 15. w=idx/3 ; coh=idx%3.
# Output: results_cv5/scaling_mo_biomor/d<w>/pnet/<coh>__response/learned_cv.json
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
WIDTHS=(96 136 192 272 352)
COHORTS=(prostate blca stad)
IDX=$SLURM_ARRAY_TASK_ID
W=${WIDTHS[$(( IDX / 3 ))]}; COH=${COHORTS[$(( IDX % 3 ))]}
echo "[cv5-scale-mo-bio] coh=$COH d=$W $(date) host=$(hostname)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.bio_redesign_curated --family pnet --cohort "$COH" \
    --task response --modes learned --cv_folds 5 --K 4 --epochs 100 --patience 15 \
    --cap_genes 3000 --d_model "$W" --device cuda \
    --out "$PROJ/results_cv5/scaling_mo_biomor/d${W}"
echo "[cv5-scale-mo-bio] DONE rc=$? $(date) coh=$COH d=$W"
