#!/usr/bin/env bash
#SBATCH --job-name=combo-bars
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:08:00
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/combo-bars-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/combo-bars-%j.out
set -uo pipefail
cd /work/mech-ai-scratch/tirtho/RecusrsiveQFormer
source /work/mech-ai-scratch/tirtho/.venv/bin/activate 2>/dev/null || true
/work/mech-ai-scratch/tirtho/.venv/bin/python scripts/make_biology_combo_bars.py
echo "[combo] DONE rc=$? $(date)"
