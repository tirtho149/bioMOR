#!/usr/bin/env bash
#SBATCH --job-name=smart-aggnet
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --array=0-2
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/aggnet-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/aggnet-%A_%a.out
# Does an AGGREGATED external bio network (STRING+KEGG+Reactome) beat no-prior on P-NET?
# Multi-modal mut+cnv, best arch (token-choice MoR, pathway markers). 3 priors x 3 seeds,
# per-seed dirs (tag has no seed). Output: results_aggnet_pnet/<prior>/s<seed>/...
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer; PY=$PROJ/../.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
TASKS=(prostate blca stad); TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "[aggnet] $TASK $(date)"; nvidia-smi -L || true
for SEED in 0 1 2; do
  for PRIOR in aggnet none coexpr; do
    echo "==== $TASK prior=$PRIOR seed=$SEED (mut_cnv, pathway, token MoR) ===="
    $PY -m recursive_marker_transformer.pathway_tasks --task "$TASK" \
        --channels mut_cnv --marker_mode pathway --recursion_mode token \
        --gene_interaction $PRIOR --n_markers 256 --d_model 128 \
        --epochs 60 --lr 3e-4 --patience 8 --seed $SEED --device cuda \
        --out results_aggnet_pnet/$PRIOR/s$SEED || echo "  prior=$PRIOR seed=$SEED FAILED"
  done
done
echo "[aggnet] DONE rc=$? $(date)"
