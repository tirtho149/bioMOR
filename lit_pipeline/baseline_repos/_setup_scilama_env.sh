set -e
VENV=/work/mech-ai-scratch/tirtho/.venv_scilama
SL=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/lit_pipeline/baseline_repos/sciLaMA
python3 -m venv "$VENV"; source "$VENV/bin/activate"; pip install -q --upgrade pip
pip install -q -e "$SL"
pip install -q scikit-learn
python - <<'PY'
import sys; sys.path.insert(0,"/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/lit_pipeline/baseline_repos/sciLaMA/src")
from sciLaMA.trainer import SciLaMATrainer
import torch; print("torch",torch.__version__,"| SciLaMATrainer import OK")
PY
echo "SCILAMA_ENV_OK"
