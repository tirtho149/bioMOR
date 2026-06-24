#!/bin/bash
#SBATCH --job-name=gf_baseline
#SBATCH --partition=nova
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/geneformer-%j.out
# Fine-tune the cancer-tuned Geneformer (V2-104M_CLcancer) on the genoNet tasks
# under the SAME seed-42 split as SMART, then regenerate the paper so the
# Geneformer macro-F1 cells in tab:foundation fill in.
set -e
cd /work/mech-ai-scratch/tirtho/RecusrsiveQFormer
mkdir -p logs
source /work/mech-ai-scratch/tirtho/.venv_geneformer/bin/activate
python lit_pipeline/baseline_repos/run_geneformer.py "$@"

# refresh the paper table with the produced numbers
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
python -m recursive_marker_transformer.make_paper --results results --outdir paper
echo "[geneformer.sbatch] done -> results_dl_baselines/Geneformer.json + paper refreshed"
