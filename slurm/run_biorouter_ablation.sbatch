#!/usr/bin/env bash
#SBATCH --job-name=cv5-biorouter
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --array=0-15%8
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/cv5-biorouter-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/cv5-biorouter-%A_%a.out
# Focused bio-router ablation on 4 datasets (Baron, Segerstolpe [SC]; STAD, BLCA [MO]),
# SAME 5-fold CV (seed 42). Four router variants (the columns):
#   none        = data-driven only router (no biology)
#   coexpr/curated = static biology only (fixed graph: co-expression [SC] / Reactome [MO])
#   learned     = learned biology graph (end-to-end)
#   learned_bio = final bioMoR (data-driven router + bio-prior learned graph)
# idx: 0-3 Baron, 4-7 Segerstolpe (SC); 8-11 STAD, 12-15 BLCA (MO). mode = idx%4.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
OUT="$PROJ/results/cv5/biorouter_ablation"

SC_MODES=(none coexpr learned learned_bio)
MO_MODES=(none curated learned learned_bio)

i=$SLURM_ARRAY_TASK_ID
m4=$(( i % 4 ))
echo "[biorouter] idx=$i host=$(hostname) $(date)"; nvidia-smi -L || true
if   [ "$i" -lt 4 ]; then
  DS=Baron;       MODE=${SC_MODES[$m4]}; KIND=sc
elif [ "$i" -lt 8 ]; then
  DS=Segerstolpe; MODE=${SC_MODES[$m4]}; KIND=sc
elif [ "$i" -lt 12 ]; then
  COH=stad;       MODE=${MO_MODES[$m4]}; KIND=mo
else
  COH=blca;       MODE=${MO_MODES[$m4]}; KIND=mo
fi

if [ "$KIND" = sc ]; then
  echo "SC bio-router: ds=$DS mode=$MODE"
  "$PY" -m recursive_marker_transformer.bio_learned_genomap --dataset "$DS" \
     --modes "$MODE" --cv_folds 5 --K 4 --epochs 100 --patience 15 \
     --n_markers 128 --device cuda --out "$OUT"
else
  echo "MO bio-router: cohort=$COH mode=$MODE"
  "$PY" -m recursive_marker_transformer.bio_redesign_curated --family pnet --cohort "$COH" \
     --task response --modes "$MODE" --cv_folds 5 --K 4 --epochs 100 --patience 15 \
     --cap_genes 3000 --device cuda --out "$OUT"
fi
echo "[biorouter] DONE idx=$i rc=$? $(date)"
