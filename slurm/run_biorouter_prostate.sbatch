#!/usr/bin/env bash
#SBATCH --job-name=cv5-biorouter-pro
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --array=0-3%4
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/cv5-biorouter-pro-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/cv5-biorouter-pro-%A_%a.out
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
MODES=(none curated learned learned_bio)
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}
echo "[biorouter-pro] mode=$MODE $(date)"; nvidia-smi -L || true
"$PY" -m recursive_marker_transformer.bio_redesign_curated --family pnet --cohort prostate \
   --task response --modes "$MODE" --cv_folds 5 --K 4 --epochs 100 --patience 15 \
   --cap_genes 3000 --device cuda --out "$PROJ/results/cv5/biorouter_ablation"
echo "[biorouter-pro] DONE rc=$? $(date)"
