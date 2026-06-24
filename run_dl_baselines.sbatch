#!/bin/bash
#SBATCH --job-name=dl_baselines
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dl_baselines-%j.out
# ---------------------------------------------------------------------------
# Reproduce the deep / foundation-model baselines on the SAME genoNet split as
# SMART, then regenerate the paper so Table tab:foundation fills in.
#
# scGeneFit runs in the main venv (CPU, no weights) and is already reproduced.
# The foundation models each need their OWN conda env (different torch/CUDA/
# flash-attn pins); activate the matching env before each block. Weights are
# already fetched under lit_pipeline/baseline_repos/<name>/.
# ---------------------------------------------------------------------------
set -e
cd /work/mech-ai-scratch/tirtho/RecusrsiveQFormer
REPOS=lit_pipeline/baseline_repos

# --- CPU baseline (ready) ---------------------------------------------------
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
python -m recursive_marker_transformer.dl_baselines --only scGeneFit

# --- sciLaMA (train-from-scratch VAE + linear probe) ------------------------
# conda activate scilama   # uv sync in $REPOS/sciLaMA
# python -m recursive_marker_transformer.dl_baselines --only sciLaMA

# --- scbenchmark (6-layer/256-d transformer baseline) -----------------------
# conda activate scbench    # see $REPOS/scbenchmark/requirements.txt (torch 2.3.1)
# python -m recursive_marker_transformer.dl_baselines --only scbenchmark

# --- scGPT (pan-cancer ckpt, fine-tune cls head) ----------------------------
# conda activate scgpt      # pip install scgpt "flash-attn<1.0.5"; CUDA 11.7
# python -m recursive_marker_transformer.dl_baselines --only scGPT

# --- Geneformer (V2-104M_CLcancer, fine-tune cell classifier) ---------------
# conda activate geneformer # pip install . in $REPOS/Geneformer (git lfs pull first)
# python -m recursive_marker_transformer.dl_baselines --only Geneformer

# --- CellPLM (20231027_85M, fine-tune cls head) -----------------------------
# conda activate cellplm    # pip install cellplm; CUDA>=11.7
# python -m recursive_marker_transformer.dl_baselines --only CellPLM

# --- Cell2Sentence (Pythia-410M, cell-sentence + LLM head) ------------------
# conda activate c2s        # pip install cell2sentence==1.1.0
# python -m recursive_marker_transformer.dl_baselines --only Cell2Sentence

# --- regenerate the paper so tab:foundation picks up the new numbers --------
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
python -m recursive_marker_transformer.make_paper --results results --outdir paper
echo "[dl_baselines] done; macro-F1 cells now filled in paper/genomicrecursiveformer.tex"
