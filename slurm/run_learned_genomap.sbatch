#!/usr/bin/env bash
#SBATCH --job-name=smart-learned-genomap
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --array=0-8
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/learned-genomap-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/learned-genomap-%A_%a.out
# FULL learned-graph sweep on the collaborator genomap single-cell suite. One array
# task per dataset; none/coexpr/random/learned x 10 seeds. Completes the "all data"
# sweep (P-NET done separately). Aggregate: bio_curated_stats has a learned panel; for
# genomap use the summary lines / a small groupby on results_learned_genomap/.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
DATASETS=(Lung Oesophagus Spleen Tcell Baron Muraro Segerstolpe Wang Xin)
DS="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
echo "[learned-genomap] start $(date) dataset=$DS"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.bio_learned_genomap \
    --dataset "$DS" --modes none coexpr random learned --seeds 0 1 2 3 4 5 6 7 8 9 \
    --K 4 --epochs 60 --device cuda
echo "[learned-genomap] DONE rc=$? $(date)"
