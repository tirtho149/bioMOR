#!/usr/bin/env bash
#SBATCH --job-name=smart-aggtcell
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/aggtcell-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/aggtcell-%j.out
# Does the aggregated external network (mouse STRING+KEGG) help on symbol-bearing
# single-cell (Tcell)? aggnet vs none vs coexpr vs learned, 5 seeds.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer; PY=$PROJ/../.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "[aggtcell] $(date)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.bio_learned_genomap --dataset Tcell \
    --modes none coexpr aggnet learned --seeds 0 1 2 3 4 --K 4 --epochs 50 \
    --device cuda --out results_aggnet_sc
echo "[aggtcell] DONE rc=$? $(date)"
