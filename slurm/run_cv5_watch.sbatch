#!/usr/bin/env bash
#SBATCH --job-name=cv5-watch
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/cv5-watch-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/cv5-watch-%j.out
# Background CPU job: every 5 min it re-renders the CV5 table/figure/ablation PNGs and
# RECOMPILES the paper PDF (real-time), until all CV5 training jobs finish. No GPU.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
cd "$PROJ"
export PYTHONUNBUFFERED=1
echo "[cv5-watch] start $(date) host=$(hostname)"
bash watch_cv5.sh
echo "[cv5-watch] end rc=$? $(date)"
