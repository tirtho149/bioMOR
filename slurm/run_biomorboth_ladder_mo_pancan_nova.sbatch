#!/usr/bin/env bash
#SBATCH --job-name=bmboth-ladder-mo-pancan
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-7%8
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/bmboth-ladder-mo-pancan-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/bmboth-ladder-mo-pancan-%A_%a.out
# NOVA (non-preemptible) version of run_biomorboth_ladder_mo_pancan: scavenger was full
# (PENDING Resources). nova needs --account=mech-ai --qos=normal. No GPU-type constraint --
# the model is tiny (d=128, batch 32) and runs on nova's rtx_6000/a40 nodes.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; source /work/mech-ai-scratch/tirtho/.venv/bin/activate 2>/dev/null || true
export PYTHONUNBUFFERED=1; export PYTHONPATH="$PROJ"; export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
# PM = pan_meta_pri (mut+CNV), PC = panmeta_response (expression)
COHS=(pan_meta_pri panmeta_response); CHS=(mut_cnv expr)
CFG_K=(2 3 2 3); CFG_R=(expert expert token token); CFG_N=(expert_k2 expert_k3 token_k2 token_k3)
i=$SLURM_ARRAY_TASK_ID
ci=$(( i % 4 )); cj=$(( i / 4 ))
coh=${COHS[$cj]}; CH=${CHS[$cj]}
K=${CFG_K[$ci]}; R=${CFG_R[$ci]}; N=${CFG_N[$ci]}
echo "[bmboth-mo-pancan-nova] coh=$coh ch=$CH cfg=$N host=$(hostname) $(date)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.pathway_tasks \
    --task "$coh" --channels "$CH" --marker_mode pathway --recursion_mode "$R" \
    --recursion_depth "$K" --gene_interaction none \
    --pathway_learned_graph --pathway_learned_fuse --bio_graph_router --pathway_pool sum \
    --n_markers 256 --d_model 128 --epochs 100 --patience 15 --batch_size 32 --lr 3e-4 \
    --cv_folds 5 --device cuda --out "results/cv5/biomor_ladder_mo/$N"
echo "[bmboth-mo-pancan-nova] DONE coh=$coh cfg=$N rc=$? $(date)"
