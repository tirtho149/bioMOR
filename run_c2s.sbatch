#!/bin/bash
#SBATCH --job-name=c2s_bl
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/c2s-%j.out
set -e
cd /work/mech-ai-scratch/tirtho/RecusrsiveQFormer
export SLURM_JOB_NAME=bash
source /work/mech-ai-scratch/tirtho/.venv_c2s/bin/activate
python lit_pipeline/baseline_repos/run_c2s.py
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
python -m recursive_marker_transformer.make_paper --results results --outdir paper
echo "[c2s.sbatch] done"
