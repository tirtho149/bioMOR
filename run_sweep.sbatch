#!/usr/bin/env bash
# ============================================================================
# SMART: Selective Marker-guided Adaptive Recursive Transformer
#        for Transcriptomic Classification
#
# Authors:
#   Koushik Howlader   - Iowa State University
#   Tirtho Roy         - Iowa State University
#   Md Tauhidul Islam  - Stanford University
#   Wei Le             - Iowa State University
#
# Copyright (c) 2026 The SMART Authors. All Rights Reserved.
#
# PROPRIETARY AND CONFIDENTIAL. Unauthorized use, copying, modification, or
# distribution of this file, in whole or in part, without the express written
# permission of the authors is STRICTLY PROHIBITED and will be prosecuted to
# the fullest extent permitted by law. See the LICENSE file for full terms.
# ============================================================================

#SBATCH --job-name=smart-sweep
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --mail-user=tirtho@iastate.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/sweep-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/sweep-%j.out
# =============================================================================
# SMART full clean GPU sweep:
#   (1) full TCGA ablation suite -> results/*.json
#   (2) single-cell generalization (4 datasets) -> results_singlecell/*.json
#   (3) regenerate the paper .tex from the fresh JSONs
#   (4) compile the PDF
# Data is already cached under data/ (no network needed on the compute node).
#   sbatch run_sweep.sbatch
# =============================================================================
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PATH="$HOME/.local/bin:$PATH"   # pdflatex / bibtex

echo "================================================================"
echo "[sweep] host=$(hostname) date=$(date)"
echo "[sweep] gpu:"; nvidia-smi -L
$PY -c "import torch;print('[sweep] torch',torch.__version__,'cuda_avail',torch.cuda.is_available(),'dev',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
echo "================================================================"

# ---- full-scale config (env-overridable) ----------------------------------
EPOCHS="${RMT_EPOCHS:-25}"
NHVG="${RMT_NHVG:-4000}"
DMODEL="${RMT_DMODEL:-128}"
NMARKERS="${RMT_NMARKERS:-256}"
DEPTH="${RMT_DEPTH:-4}"

# ---- (1) TCGA full ablation suite -----------------------------------------
echo "[sweep] STEP 1/4: TCGA suite (epochs=$EPOCHS n_hvg=$NHVG d_model=$DMODEL n_markers=$NMARKERS depth=$DEPTH)"
$PY -m recursive_marker_transformer.experiments \
  --epochs "$EPOCHS" --n_hvg "$NHVG" --d_model "$DMODEL" \
  --n_markers "$NMARKERS" --depth "$DEPTH" --heads cancer_type \
  --outdir results
TCGA_RC=$?
echo "[sweep] STEP 1 exit=$TCGA_RC"

# ---- (2) single-cell generalization ---------------------------------------
echo "[sweep] STEP 2/4: single-cell (4 datasets)"
$PY -m recursive_marker_transformer.singlecell \
  --epochs 25 --batch_size 256 --lr 1e-3 --patience 6 \
  --d_model 96 --n_markers 128 --device cuda \
  --out results_singlecell
SC_RC=$?
echo "[sweep] STEP 2 exit=$SC_RC"

# ---- (3) regenerate paper .tex --------------------------------------------
echo "[sweep] STEP 3/4: make_paper"
$PY -m recursive_marker_transformer.make_paper --results results --outdir paper

# ---- (4) compile PDF -------------------------------------------------------
echo "[sweep] STEP 4/4: compile PDF"
if command -v pdflatex >/dev/null 2>&1; then
  ( cd paper
    pdflatex -interaction=nonstopmode genomicrecursiveformer.tex >/dev/null 2>&1 || true
    bibtex genomicrecursiveformer >/dev/null 2>&1 || true
    pdflatex -interaction=nonstopmode genomicrecursiveformer.tex >/dev/null 2>&1 || true
    pdflatex -interaction=nonstopmode genomicrecursiveformer.tex >/dev/null 2>&1 || true )
  echo "[sweep] PDF -> paper/genomicrecursiveformer.pdf"
else
  echo "[sweep] WARNING: pdflatex not found; .tex generated, compile elsewhere"
fi

echo "[sweep] DONE date=$(date)  (tcga_rc=$TCGA_RC sc_rc=$SC_RC)"
