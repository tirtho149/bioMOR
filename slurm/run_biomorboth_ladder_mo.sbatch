#!/usr/bin/env bash
#SBATCH --job-name=bmboth-ladder-mo
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --array=0-19%20
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/bmboth-ladder-mo-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/bmboth-ladder-mo-%A_%a.out
# Canonical bioMoR (bio_both) across the ladder for Table 2, multi-omics, using the
# PROVIDED Reactome graph (adjacency_matrix.csv) in pathway space. 5 (K,routing) x 4 cohorts.
# Identical protocol (pathway, m256, d128, batch32, lr3e-4, e100, p15, sum), CV5 seed42.
# expert_k4 = run_injection_mo cond=both (results/cv5/inject_mo/both).
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; source /work/mech-ai-scratch/tirtho/.venv/bin/activate 2>/dev/null || true
export PYTHONUNBUFFERED=1; export PYTHONPATH="$PROJ"; export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
COHS=(prostate blca stad brca)
CFG_K=(2 3 2 3 4); CFG_R=(expert expert token token token); CFG_N=(expert_k2 expert_k3 token_k2 token_k3 token_k4)
i=$SLURM_ARRAY_TASK_ID
coh=${COHS[$(( i % 4 ))]}; ci=$(( i / 4 ))
K=${CFG_K[$ci]}; R=${CFG_R[$ci]}; N=${CFG_N[$ci]}
CH=mut_cnv; [ "$coh" = brca ] && CH=mut
echo "[bmboth-mo] coh=$coh cfg=$N host=$(hostname) $(date)"; nvidia-smi -L || true
# REDESIGNED both = provided pathway graph on BOTH sites (embedding smoothing + fuse +
# zero-init graph-conv router), NO static centrality prior -- identical to inject_mo/both.
$PY -m recursive_marker_transformer.pathway_tasks \
    --task "$coh" --channels "$CH" --marker_mode pathway --recursion_mode "$R" \
    --recursion_depth "$K" --gene_interaction none \
    --pathway_learned_graph --pathway_learned_fuse --bio_graph_router --pathway_pool sum \
    --n_markers 256 --d_model 128 --epochs 100 --patience 15 --batch_size 32 --lr 3e-4 \
    --cv_folds 5 --device cuda --out "results/cv5/biomor_ladder_mo/$N"
echo "[bmboth-mo] DONE coh=$coh cfg=$N rc=$? $(date)"
