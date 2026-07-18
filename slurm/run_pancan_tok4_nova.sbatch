#!/usr/bin/env bash
#SBATCH --job-name=bmboth-tok4-pancan
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/bmboth-tok4-pancan-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/bmboth-tok4-pancan-%A_%a.out
# The 2 missing Table-2 cells: PM/PC at token_k4 (the +Token choice bioMoR row). The pancan
# ladder ran expert_k2/k3 + token_k2/k3 but not token_k4; only 3M got token_k4.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer; PY=$PROJ/../.venv/bin/python
cd "$PROJ"; source /work/mech-ai-scratch/tirtho/.venv/bin/activate 2>/dev/null || true
export PYTHONUNBUFFERED=1 PYTHONPATH="$PROJ" OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
COHS=(pan_meta_pri panmeta_response); CHS=(mut_cnv expr)
i=$SLURM_ARRAY_TASK_ID; coh=${COHS[$i]}; CH=${CHS[$i]}
echo "[tok4-pancan] coh=$coh $(date)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.pathway_tasks \
    --task "$coh" --channels "$CH" --marker_mode pathway --recursion_mode token --recursion_depth 4 \
    --gene_interaction none --pathway_learned_graph --pathway_learned_fuse --bio_graph_router \
    --pathway_pool sum --n_markers 256 --d_model 128 --epochs 100 --patience 15 --batch_size 32 \
    --lr 3e-4 --cv_folds 5 --device cuda --out "results/cv5/biomor_ladder_mo/token_k4"
echo "[tok4-pancan] DONE coh=$coh rc=$? $(date)"
