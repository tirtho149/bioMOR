#!/usr/bin/env bash
#SBATCH --job-name=smart-learned-smoke
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/learned-smoke-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/learned-smoke-%j.out
# GPU decision run: does the DATA-DRIVEN learned graph beat the plain router and the
# biology graphs? prostate (the cohort where a fixed graph helped most vs none),
# none/curated/random/learned x 3 seeds, full 60 epochs. Gate the full sweep on this.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "[learned-smoke] start $(date)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.bio_redesign_curated \
    --family pnet --cohort prostate --modes none curated random learned \
    --seeds 0 1 2 --K 4 --epochs 60 --cap_genes 3000 --device cuda \
    --out "$PROJ/results_learned_smoke"
echo "[learned-smoke] DONE rc=$? $(date)"
