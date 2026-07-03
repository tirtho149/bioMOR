#!/usr/bin/env bash
#SBATCH --job-name=smart-learned-pnet
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/learned-pnet-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/learned-pnet-%A_%a.out
# FULL learned-graph sweep on all P-NET cohorts (collaborator data). One array task
# per cohort; none/curated/random/learned x 10 seeds. none/curated/random already
# exist from job 11415601 and are SKIPPED (reused); only `learned` is computed fresh.
# Aggregate: python -m recursive_marker_transformer.bio_curated_stats
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
COHORTS=(prostate blca stad brca)
COHORT="${COHORTS[$SLURM_ARRAY_TASK_ID]}"
echo "[learned-pnet] start $(date) cohort=$COHORT"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.bio_redesign_curated \
    --family pnet --cohort "$COHORT" --task response \
    --modes none curated random learned --seeds 0 1 2 3 4 5 6 7 8 9 \
    --K 4 --epochs 60 --cap_genes 3000 --device cuda
echo "[learned-pnet] DONE rc=$? $(date)"
