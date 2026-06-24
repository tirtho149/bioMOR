#!/bin/bash
# Isolated env for Geneformer (needs transformers <5; SMART venv has 5.x).
set -e
VENV=/work/mech-ai-scratch/tirtho/.venv_geneformer
GF=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/lit_pipeline/baseline_repos/Geneformer

python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install -q --upgrade pip

# torch with CUDA 12.8 wheels (matches the nova GPU nodes)
pip install -q torch --index-url https://download.pytorch.org/whl/cu128

# Geneformer-compatible stack (transformers 4.x line)
pip install -q "transformers==4.44.2" "datasets==2.20.0" "accelerate==0.33.0" \
               "peft==0.12.0" scikit-learn scipy pandas numpy pyarrow \
               anndata scanpy loompy tdigest statsmodels

# Geneformer itself without letting it re-resolve transformers
pip install -q --no-deps -e "$GF"

python - <<'PY'
import transformers, torch
print("transformers", transformers.__version__, "| torch", torch.__version__)
from geneformer import Classifier, TranscriptomeTokenizer
print("Geneformer import OK (Classifier + TranscriptomeTokenizer)")
PY
echo "GENEFORMER_ENV_OK"
