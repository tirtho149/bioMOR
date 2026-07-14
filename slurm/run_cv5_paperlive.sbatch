#!/usr/bin/env bash
#SBATCH --job-name=cv5-paperlive
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/cv5-paperlive-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/cv5-paperlive-%j.out
# CPU background job: real-time regenerate native LaTeX tables + recompile the paper PDF
# from results_cv5/, every ~3 min for 24h. No GPU.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
cd "$PROJ"; export PYTHONUNBUFFERED=1
echo "[cv5-paperlive] start $(date) host=$(hostname)"
bash paper_live.sh
echo "[cv5-paperlive] end rc=$? $(date)"
