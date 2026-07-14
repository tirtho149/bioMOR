#!/usr/bin/env bash
#SBATCH --job-name=scalewatch
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/scalewatch-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/scalewatch-%j.out
# Rebuilds the scaling figure every 90s as results land; exits when no 'scale-' jobs remain.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"
for i in $(seq 1 400); do
  $PY build_scaling_figure.py > /dev/null 2>> logs/watch_scaling.log
  n=$(squeue -u tirtho -h -o "%j" 2>/dev/null | grep -c 'scale-')
  echo "[scalewatch $(date +%H:%M:%S)] rebuilt figure; $n scale jobs left" >> logs/watch_scaling.log
  if [ "$n" -eq 0 ]; then
    $PY build_scaling_figure.py > /dev/null 2>> logs/watch_scaling.log
    echo "[scalewatch $(date +%H:%M:%S)] ALL DONE — final figure written, exiting" >> logs/watch_scaling.log
    break
  fi
  sleep 90
done
