#!/bin/bash
#SBATCH --job-name=sct_full
#SBATCH --account=mech-ai-scavenger --qos=scavenger --partition=scavenger
#SBATCH --gres=gpu:1 --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8 --mem=96G --time=08:00:00 --requeue
#SBATCH --array=0-6
#SBATCH --output=/work/mech-ai-scratch/tirtho/bioMOR-baselines/logs/sct_full_%A_%a.out
set -euo pipefail
source /work/mech-ai-scratch/tirtho/.venvs_baselines/scTransSort/bin/activate
export BIOMOR_ROOT=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
export TF_CPP_MIN_LOG_LEVEL=2
cd /work/mech-ai-scratch/tirtho/bioMOR-baselines
DS=(Baron Lung Oesophagus Segerstolpe Spleen Tcell Xin)
D=${DS[$SLURM_ARRAY_TASK_ID]}
echo "=== scTransSort full run dataset=$D ==="
python sc_baselines/scTransSort/scripts/sctranssort_cv.py --dataset "$D" --folds 5 --epochs 40
echo "FULL DONE $D"
