#!/bin/bash
#SBATCH --job-name=cellplm_bl
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/cellplm-%j.out
set -e
cd /work/mech-ai-scratch/tirtho/RecusrsiveQFormer
export SLURM_JOB_NAME=bash
source /work/mech-ai-scratch/tirtho/.venv_cellplm/bin/activate
python lit_pipeline/baseline_repos/run_cellplm.py
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
python -m recursive_marker_transformer.make_paper --results results --outdir paper
echo "[cellplm.sbatch] done"
