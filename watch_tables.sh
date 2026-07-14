#!/usr/bin/env bash
# Real-time table refresher: rebuilds the ladder + ablation PNGs every 90s as SLURM
# results land, and exits once no biomor/pancan jobs remain in the queue.
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/usr/bin/python
cd "$PROJ"
for i in $(seq 1 240); do   # up to 6 h
  $PY build_ladder_table.py  >/dev/null 2>>logs/watch_tables.log
  $PY build_ablation_table.py >/dev/null 2>>logs/watch_tables.log
  n=$(squeue -u tirtho -h -o "%j" 2>/dev/null | grep -c -E 'biomor|pancan')
  echo "[watch $(date +%H:%M:%S)] rebuilt; $n biomor/pancan jobs still queued/running" >> logs/watch_tables.log
  if [ "$n" -eq 0 ]; then
    $PY build_ladder_table.py  >/dev/null 2>>logs/watch_tables.log
    $PY build_ablation_table.py >/dev/null 2>>logs/watch_tables.log
    echo "[watch $(date +%H:%M:%S)] all jobs done — final rebuild written, exiting" >> logs/watch_tables.log
    break
  fi
  sleep 90
done
