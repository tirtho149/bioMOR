#!/bin/bash
#SBATCH --job-name=scilama_bl
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/scilama-%j.out
set -e
cd /work/mech-ai-scratch/tirtho/RecusrsiveQFormer
# stop PyTorch-Lightning from auto-binding to the SLURM cluster env (it rejects
# the run over SLURM_NTASKS); "bash"/"interactive" makes Lightning treat it as
# an unmanaged single-process run.
export SLURM_JOB_NAME=bash
source /work/mech-ai-scratch/tirtho/.venv_scilama/bin/activate
python lit_pipeline/baseline_repos/run_scilama.py
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
python -m recursive_marker_transformer.make_paper --results results --outdir paper
echo "[scilama.sbatch] done"
