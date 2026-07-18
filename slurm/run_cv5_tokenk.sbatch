#!/usr/bin/env bash
#SBATCH --job-name=cv5-tokenk
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --array=0-47%7
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/cv5-tokenk-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/cv5-tokenk-%A_%a.out
# Token-choice MoR-general and bioMoR at recursion depth K in {2,3}, all datasets,
# SAME 5-fold CV (seed 42) as Table 2. Fills the 4 new token rows.
#   idx 0-15  : MoR-general SC  token  (8 ds x K{2,3})
#   idx 16-25 : MoR-general MO  token  (5 tasks x K{2,3})
#   idx 26-41 : bioMoR      SC  token  (8 ds x K{2,3})
#   idx 42-47 : bioMoR      MO  token  (3 cohorts x K{2,3})
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

SC_DS=(baron lung muraro oesophagus segerstolpe spleen tcell xin)
BIO_DS=(Baron Lung Muraro Oesophagus Segerstolpe Spleen Tcell Xin)
MO_T=(prostate blca stad pan_meta_pri panmeta_response)
MO_CH=(mut_cnv mut_cnv mut_cnv mut_cnv expr)
PNC=(prostate blca stad)
KS=(2 3)

i=$SLURM_ARRAY_TASK_ID
echo "[tokenk] idx=$i host=$(hostname) $(date)"; nvidia-smi -L || true
if   [ "$i" -lt 16 ]; then
  k=${KS[$(( i / 8 ))]}; ds=${SC_DS[$(( i % 8 ))]}
  echo "MoR-gen SC token k=$k ds=$ds"
  "$PY" -m recursive_marker_transformer.singlecell --data data/singlecell \
     --out "results/cv5/sc/token_k${k}" --datasets "$ds" --cv_folds 5 --seed 42 \
     --epochs 100 --d_model 96 --n_markers 128 --batch_size 128 --lr 0.001 \
     --weight_decay 0.00001 --patience 15 --device cuda \
     --recursion_mode token --recursion_depth "${k}"
elif [ "$i" -lt 26 ]; then
  j=$(( i - 16 )); k=${KS[$(( j / 5 ))]}; t=${MO_T[$(( j % 5 ))]}; ch=${MO_CH[$(( j % 5 ))]}
  echo "MoR-gen MO token k=$k task=$t chan=$ch"
  "$PY" -m recursive_marker_transformer.pathway_tasks --task "$t" --channels "$ch" \
     --marker_mode pathway --n_markers 256 --d_model 128 --epochs 100 --lr 3e-4 \
     --patience 15 --batch_size 32 --cv_folds 5 --device cuda \
     --out "results/cv5/mo/token_k${k}" --gene_interaction reactome \
     --recursion_mode token --recursion_depth "${k}"
elif [ "$i" -lt 42 ]; then
  j=$(( i - 26 )); k=${KS[$(( j / 8 ))]}; ds=${BIO_DS[$(( j % 8 ))]}
  echo "bioMoR SC token k=$k ds=$ds"
  "$PY" -m recursive_marker_transformer.bio_learned_genomap --dataset "$ds" \
     --modes learned --cv_folds 5 --K "${k}" --epochs 100 --patience 15 \
     --n_markers 128 --device cuda --recursion_mode token \
     --out "results/cv5/biomor_sc_token_k${k}"
else
  j=$(( i - 42 )); k=${KS[$(( j / 3 ))]}; c=${PNC[$(( j % 3 ))]}
  echo "bioMoR MO token k=$k cohort=$c"
  "$PY" -m recursive_marker_transformer.bio_redesign_curated --family pnet --cohort "$c" \
     --task response --modes learned --cv_folds 5 --K "${k}" --epochs 100 --patience 15 \
     --cap_genes 3000 --device cuda --recursion_mode token \
     --out "results/cv5/biomor_mo_token_k${k}"
fi
echo "[tokenk] DONE idx=$i rc=$? $(date)"
