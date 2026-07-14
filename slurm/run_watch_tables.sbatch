#!/usr/bin/env bash
#SBATCH --job-name=tblwatch
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/tblwatch-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/tblwatch-%j.out
# Real-time table refresher (runs on a compute node so `sleep` is allowed).
# Rebuilds ladder + ablation PNGs every 60s as SLURM results land; exits when the
# biomor/pancan training jobs are all gone.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"
for i in $(seq 1 360); do
  $PY build_ladder_table.py   > /dev/null 2>> logs/watch_tables.log
  $PY build_ablation_table.py > /dev/null 2>> logs/watch_tables.log
  n=$(squeue -u tirtho -h -o "%j" 2>/dev/null | grep -c -E 'biomor|pancan|e100')
  echo "[tblwatch $(date +%H:%M:%S)] rebuilt PNGs; $n biomor/pancan jobs left" >> logs/watch_tables.log
  if [ "$n" -eq 0 ]; then
    $PY build_ladder_table.py   > /dev/null 2>> logs/watch_tables.log
    $PY build_ablation_table.py > /dev/null 2>> logs/watch_tables.log
    echo "[tblwatch $(date +%H:%M:%S)] ALL DONE — final PNGs written, exiting" >> logs/watch_tables.log
    break
  fi
  sleep 60
done
