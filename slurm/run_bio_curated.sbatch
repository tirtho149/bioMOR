#!/usr/bin/env bash
#SBATCH --job-name=smart-bio-curated
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --array=0-19
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/bio-curated-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/bio-curated-%A_%a.out
# CURATED-NETWORK bio-router falsification (BIO_ROUTER_REDESIGN.txt) on the
# symbol-bearing cohorts. One array task per (family, cohort, task) cell; each runs
# none/curated/random x 10 seeds. Missing/underpopulated cells self-SKIP. Aggregate
# with:  python -m recursive_marker_transformer.bio_curated_stats
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p logs

# (family cohort task) cells. TCGA stage/T/N/os_binary + P-NET response.
CELLS=(
  "tcga breast pathologic_stage"  "tcga breast pathologic_T"  "tcga breast pathologic_N"  "tcga breast os_binary"
  "tcga lung pathologic_stage"    "tcga lung pathologic_T"    "tcga lung pathologic_N"    "tcga lung os_binary"
  "tcga head_neck pathologic_stage" "tcga head_neck pathologic_T" "tcga head_neck pathologic_N" "tcga head_neck os_binary"
  "tcga thyroid pathologic_stage" "tcga thyroid pathologic_T" "tcga thyroid pathologic_N" "tcga thyroid os_binary"
  "pnet prostate response"  "pnet blca response"  "pnet stad response"  "pnet brca response"
)
CELL="${CELLS[$SLURM_ARRAY_TASK_ID]}"
read -r FAMILY COHORT TASK <<< "$CELL"
echo "[bio-curated] start $(date) cell='$CELL'"; nvidia-smi -L || true

$PY -m recursive_marker_transformer.bio_redesign_curated \
    --family "$FAMILY" --cohort "$COHORT" --task "$TASK" \
    --modes none curated random --seeds 0 1 2 3 4 5 6 7 8 9 \
    --K 4 --epochs 60 --cap_genes 3000 --device cuda
echo "[bio-curated] DONE rc=$? $(date)"
