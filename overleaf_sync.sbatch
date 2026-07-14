#!/usr/bin/env bash
#SBATCH --job-name=overleaf-sync
#SBATCH --partition=nova
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/overleaf_sync_%j.log
# ---------------------------------------------------------------------------
# Durable background Overleaf <-> paper/ sync. Runs overleaf_sync.sh (3s poll
# bidirectional git-bridge sync) for just under the walltime, then RESUBMITS
# ITSELF so the sync never stops. Needs only git + rsync (no venv / no GPU).
#
#   start :  sbatch overleaf_sync.sbatch
#   stop  :  touch .overleaf_sync_stop   (then scancel the running job)
#            -> the trailing self-resubmit is skipped, so the chain ends.
# ---------------------------------------------------------------------------
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
cd "$PROJ" || exit 1
mkdir -p logs

STOP_FLAG="$PROJ/.overleaf_sync_stop"
# run for ~6.9 days, leaving headroom before the 7-day wall for a clean resubmit
RUN_SECONDS=$(( 7*24*3600 - 2*3600 ))

if [ -f "$STOP_FLAG" ]; then
  echo "[sbatch] stop flag present ($STOP_FLAG); not starting. Remove it to re-enable."
  exit 0
fi

echo "[sbatch] job $SLURM_JOB_ID on $(hostname) starting sync for ${RUN_SECONDS}s"
timeout --signal=TERM "${RUN_SECONDS}s" bash "$PROJ/overleaf_sync.sh"
rc=$?
echo "[sbatch] sync exited rc=$rc after time budget"

# self-resubmit to keep the sync alive, unless asked to stop
if [ -f "$STOP_FLAG" ]; then
  echo "[sbatch] stop flag present; chain ends here."
else
  echo "[sbatch] resubmitting self to continue sync..."
  sbatch "$PROJ/overleaf_sync.sbatch"
fi
