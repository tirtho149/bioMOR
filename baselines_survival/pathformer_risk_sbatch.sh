#!/bin/bash
#SBATCH --job-name=pf_risk
#SBATCH --partition=nova
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/lustre/hdd/LAS/weile-lab/howlader/GraphPath_baselines/baselines_survival/pathformer_risk.log
#SBATCH --error=/lustre/hdd/LAS/weile-lab/howlader/GraphPath_baselines/baselines_survival/pathformer_risk.log

DIR=/lustre/hdd/LAS/weile-lab/howlader/GraphPath_baselines/baselines_survival
PATH_PY=/lustre/hdd/LAS/weile-lab/howlader/envs/path/bin/python
cd "$DIR" || exit 1
export PYTHONPATH="$DIR:$PYTHONPATH"

echo "############## GPU INFO ##############"; nvidia-smi -L 2>&1 | head
echo "### START Pathformer risk export"; t0=$(date +%s)
PYTHONPATH="$DIR" $PATH_PY -u pathformer_survival.py
rc=$?; t1=$(date +%s)
echo "### END Pathformer exit=$rc elapsed=$((t1-t0))s"
