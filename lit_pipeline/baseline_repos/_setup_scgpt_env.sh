set -e
VENV=/work/mech-ai-scratch/tirtho/.venv_scgpt
python3 -m venv "$VENV"; source "$VENV/bin/activate"; pip install -q --upgrade pip
pip install -q torch --index-url https://download.pytorch.org/whl/cu128
pip install -q scgpt scikit-learn scanpy anndata 2>&1 | tail -3 || true
python - <<'PY'
try:
    import scgpt, torch; print("scgpt OK | torch", torch.__version__)
except Exception as e:
    import traceback; traceback.print_exc()
PY
echo "SCGPT_ENV_DONE"
