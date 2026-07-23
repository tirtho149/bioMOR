#!/bin/bash
#SBATCH --job-name=sct_venv
#SBATCH --account=mech-ai-scavenger --qos=scavenger --partition=scavenger
#SBATCH --gres=gpu:1 --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8 --mem=64G --time=02:00:00 --requeue
#SBATCH --output=/work/mech-ai-scratch/tirtho/bioMOR-baselines/logs/sct_venv_%j.out
set -euo pipefail
VENV=/work/mech-ai-scratch/tirtho/.venvs_baselines/scTransSort
mkdir -p /work/mech-ai-scratch/tirtho/.venvs_baselines
if [ ! -f "$VENV/bin/activate" ]; then
  echo "creating venv $VENV"
  python -m venv --system-site-packages "$VENV"
fi
source "$VENV/bin/activate"
pip install --upgrade pip
# Self-sufficient TF venv: CUDA-enabled tensorflow + the data stack the runner needs.
pip install "tensorflow[and-cuda]==2.15.1" "numpy<2.0" scanpy anndata scikit-learn pandas
python -c "import tensorflow as tf; print('TF', tf.__version__); print('GPUs', tf.config.list_physical_devices('GPU'))"
python -c "import scanpy, anndata, sklearn; print('scanpy/anndata/sklearn OK')"
echo "VENV SETUP DONE"
