#!/bin/bash
# Full 5-fold training for every survival baseline in 3-MODAL mode
# (CNV + mutation + gene expression, pan_survival_3mod: 8454 patients,
# 17940 common genes). Writes to results_3mod_smoke/ so the 2-modal results in
# results/ are preserved. Torch baselines use `path`; PathCNN (Keras) uses `pnet`.
set +e
DIR=/lustre/hdd/LAS/weile-lab/howlader/GraphPath_baselines/baselines_survival
ENVS=/lustre/hdd/LAS/weile-lab/howlader/envs
PATH_PY=$ENVS/path/bin/python
PNET_PY=$ENVS/pnet/bin/python
cd "$DIR" || exit 1
export PYTHONPATH="$DIR:$PYTHONPATH"
export SURV_3MODAL=1
export SURV_RESULTS_DIR=results_3mod_smoke
export SURV_DATASET_TAG=pan_survival_3mod
mkdir -p "$DIR/results_3mod_smoke"

echo "############## 3-MODAL SMOKE (CNV+mutation+expression) ##############"
echo "SURV_3MODAL=$SURV_3MODAL  RESULTS_DIR=$SURV_RESULTS_DIR"
nvidia-smi -L 2>&1 | head

export SMOKE=1
run() {
  name="$1"; shift
  echo ""; echo "### START $name"; echo "### CMD: $*"
  t0=$(date +%s); "$@"; rc=$?; t1=$(date +%s)
  echo "### END $name exit=$rc elapsed=$((t1-t0))s"; echo "RESULT $name $rc"
}

run "CNN_ei"     $PATH_PY -u cnn_ei_survival.py
run "CNN_li"     $PATH_PY -u cnn_li_survival.py
run "MOGONET"    $PATH_PY -u mogonet_survival.py
run "MOGAT"      $PATH_PY -u mogat_survival.py
run "Pathformer" $PATH_PY -u pathformer_survival.py
run "pnet"       $PATH_PY -u pnet_survival.py
run "PathCNN"    $PNET_PY -u pathcnn_survival.py

echo ""; echo "############## ALL 3-MODAL SMOKE RUNS DONE ##############"
