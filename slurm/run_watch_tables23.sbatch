#!/usr/bin/env bash
#SBATCH --job-name=watch-tab23
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/watch-tab23-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/watch-tab23-%j.out
# Dedicated dynamic watcher for TABLE 2 (scripts/build_cv5_tex.py -> paper/cv5_main_table.tex) and
# TABLE 3 (scripts/build_injection_table.py -> paper/cv5_injection_table.tex). Regenerates both from
# results/cv5/ every 60s so pending cells show 'run...' and fill live as CV folds land; the
# overleaf-sync job then pushes them. Runs on a compute node (sleep allowed). Exits only after
# the bioMoR/injection training jobs have been gone for 5 consecutive checks (avoids premature
# exit on transient empty queues / scavenger requeue).
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; source /work/mech-ai-scratch/tirtho/.venv/bin/activate 2>/dev/null || true
export PYTHONUNBUFFERED=1; export PYTHONPATH="$PROJ"
LOG=logs/watch_tables23.log
gone=0
for i in $(seq 1 1440); do
  $PY scripts/build_cv5_tex.py        >> "$LOG" 2>&1 || echo "[watch23 $(date +%H:%M:%S)] build_cv5_tex ERROR" >> "$LOG"
  $PY scripts/build_injection_table.py >> "$LOG" 2>&1 || echo "[watch23 $(date +%H:%M:%S)] build_injection_table ERROR" >> "$LOG"
  # progress: count filled bio_both cells (SC + MO)
  sc=$(ls results/cv5/biomor_canonical/*/bio_both_cv.json 2>/dev/null | wc -l)
  mo=$(ls results/cv5/inject_mo/both/*_cv.json 2>/dev/null | wc -l)
  n=$(squeue -u tirtho -h -o "%j" 2>/dev/null | grep -cE 'inject|bmboth')
  echo "[watch23 $(date +%H:%M:%S)] Tab2+Tab3 regenerated; SC both=$sc MO both=$mo; $n training jobs left" >> "$LOG"
  if [ "$n" -eq 0 ]; then gone=$((gone+1)); else gone=0; fi
  if [ "$gone" -ge 5 ]; then
    $PY scripts/build_cv5_tex.py >> "$LOG" 2>&1; $PY scripts/build_injection_table.py >> "$LOG" 2>&1
    echo "[watch23 $(date +%H:%M:%S)] ALL DONE — final Table 2 & 3 written, exiting" >> "$LOG"
    break
  fi
  sleep 60
done
