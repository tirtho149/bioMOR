#!/usr/bin/env bash
#SBATCH --job-name=prostate-panels
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/prostate-panels-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/prostate-panels-%j.out
set -uo pipefail
cd /work/mech-ai-scratch/tirtho/RecusrsiveQFormer
source /work/mech-ai-scratch/tirtho/.venv/bin/activate 2>/dev/null || true
export PYTHONPATH=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer PYTHONUNBUFFERED=1
/work/mech-ai-scratch/tirtho/.venv/bin/python scripts/prostate_depth_panels.py
echo "rc=$?"
