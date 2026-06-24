#!/bin/bash
# Build the 3 remaining baseline venvs SEQUENTIALLY (no parallel torch installs).
# Each step is best-effort; failures are logged but don't abort the rest.
set -u
BR=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/lit_pipeline/baseline_repos
PIPQ="pip install -q"

echo "########## [1/3] scbench (torch + numpy/pandas/sklearn) ##########"
V=/work/mech-ai-scratch/tirtho/.venv_scbench
python3 -m venv "$V" && source "$V/bin/activate" && $PIPQ --upgrade pip
$PIPQ torch numpy pandas scikit-learn scipy 2>&1 | tail -3
python -c "import torch,sys; sys.path.insert(0,'$BR/scbenchmark'); from models_utils.model import scModel; print('scbench OK | torch', torch.__version__)" 2>&1 | tail -5
deactivate

echo "########## [2/3] c2s (cell2sentence -e: torch/transformers/datasets/scanpy) ##########"
V=/work/mech-ai-scratch/tirtho/.venv_c2s
python3 -m venv "$V" && source "$V/bin/activate" && $PIPQ --upgrade pip
$PIPQ -e "$BR/Cell2Sentence" 2>&1 | tail -3
$PIPQ scikit-learn 2>&1 | tail -2
python -c "import sys; sys.path.insert(0,'$BR/Cell2Sentence/src'); import cell2sentence as cs, torch, transformers; print('c2s OK | torch', torch.__version__, '| tf', transformers.__version__)" 2>&1 | tail -5
deactivate

echo "########## [3/3] cellplm (torch + scanpy/anndata/sklearn/tqdm) ##########"
V=/work/mech-ai-scratch/tirtho/.venv_cellplm
python3 -m venv "$V" && source "$V/bin/activate" && $PIPQ --upgrade pip
$PIPQ torch scanpy anndata scikit-learn tqdm scipy einops torchmetrics 2>&1 | tail -3
python -c "import sys; sys.path.insert(0,'$BR/CellPLM'); from CellPLM.pipeline.cell_embedding import CellEmbeddingPipeline; import torch; print('cellplm OK | torch', torch.__version__)" 2>&1 | tail -8
deactivate

echo "ALL_ENV_BUILDS_DONE"
