#!/usr/bin/env bash
#SBATCH --job-name=tex-refresh
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/tex-refresh-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/tex-refresh-%j.out
# Regenerate ALL paper .tex table fragments (Table 2 = cv5_main_table, NEW Table 3 =
# cv5_injection_table, appendix posf1, ablation) from results/cv5/ every 90s so the
# Overleaf-synced paper shows 'run...' cells that fill as CV folds land. Runs on a
# compute node (sleep allowed); exits when the bioMoR/inject training jobs are gone.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
cd "$PROJ"; source /work/mech-ai-scratch/tirtho/.venv/bin/activate 2>/dev/null || true
for i in $(seq 1 400); do
  bash scripts/refresh_cv5.sh >> logs/tex_refresh.log 2>&1 || echo "[refresh $(date +%H:%M:%S)] refresh_cv5 error" >> logs/tex_refresh.log
  n=$(squeue -u tirtho -h -o "%j" 2>/dev/null | grep -c -E 'canon-biomor|inject-sc|inject-mo|bmboth|3m-cv5')
  echo "[refresh $(date +%H:%M:%S)] regenerated .tex; $n training jobs left" >> logs/tex_refresh.log
  [ "$n" -eq 0 ] && { bash scripts/refresh_cv5.sh >> logs/tex_refresh.log 2>&1; echo "[refresh] ALL DONE" >> logs/tex_refresh.log; break; }
  sleep 90
done
