#!/bin/bash
# Full 5-fold training for every survival baseline (no SMOKE: all folds, full
# epochs), one by one, on one GPU. Torch baselines use the `path` env; PathCNN
# (Keras) uses `pnet`. Failures don't abort the rest.
set +e
DIR=/lustre/hdd/LAS/weile-lab/howlader/GraphPath_baselines/baselines_survival
ENVS=/lustre/hdd/LAS/weile-lab/howlader/envs
PATH_PY=$ENVS/path/bin/python
PNET_PY=$ENVS/pnet/bin/python
cd "$DIR" || exit 1
export PYTHONPATH="$DIR:$PYTHONPATH"

echo "############## GPU INFO ##############"; nvidia-smi -L 2>&1 | head

run() {
  name="$1"; shift
  echo ""; echo "### START $name"; echo "### CMD: $*"
  t0=$(date +%s); PYTHONPATH="$DIR" "$@"; rc=$?; t1=$(date +%s)
  echo "### END $name exit=$rc elapsed=$((t1-t0))s"; echo "RESULT $name $rc"
}

run "CNN_ei"     $PATH_PY -u cnn_ei_survival.py
run "CNN_li"     $PATH_PY -u cnn_li_survival.py
run "MOGONET"    $PATH_PY -u mogonet_survival.py
run "MOGAT"      $PATH_PY -u mogat_survival.py
run "Pathformer" $PATH_PY -u pathformer_survival.py
run "pnet"       $PATH_PY -u pnet_survival.py
run "PathCNN"    $PNET_PY -u pathcnn_survival.py

echo ""; echo "############## ALL SURVIVAL FULL RUNS DONE ##############"
