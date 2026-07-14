#!/usr/bin/env bash
# Refresh the three CV5 artifacts + recompile the paper PDF every 5 min until all
# CV5 TRAINING jobs finish (excludes this watch job itself). Meant to run as a SLURM
# job (run_cv5_watch.sbatch) so it survives login-node logout.
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
export PATH="$HOME/.local/bin:$PATH"      # user-local pdflatex
cd "$PROJ"
for i in $(seq 1 288); do
  ./refresh_cv5.sh > logs/cv5_refresh.log 2>&1
  n=$(squeue -u tirtho -h -o '%j' -t RUNNING,PENDING | grep cv5 | grep -v cv5-watch | wc -l)
  echo "[watch $(date +%m-%d_%H:%M)] cv5 training jobs left=$n" >> logs/cv5_watch.log
  if [ "$n" -eq 0 ]; then echo "ALL CV5 JOBS DONE $(date)" >> logs/cv5_watch.log; break; fi
  sleep 300
done
./refresh_cv5.sh > logs/cv5_refresh.log 2>&1
echo "[watch] final refresh done $(date)" >> logs/cv5_watch.log
