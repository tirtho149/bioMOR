#!/usr/bin/env bash
# Real-time table + paper updater (CPU). Every ~3 min: regenerate the native LaTeX
# table fragments from results_cv5/ and recompile genomicrecursiveformer.pdf. Runs the
# FULL walltime regardless of queue state (does NOT exit on a momentarily-empty queue),
# so the tables/PDF stay live for the whole rerun. Meant to run as a SLURM CPU job.
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
export PATH="$HOME/.local/bin:$PATH"          # user-local pdflatex
cd "$PROJ"
LOG=logs/cv5_paperlive.log
for i in $(seq 1 480); do          # 480 * 180s ~= 24h
  # 1) native LaTeX table fragments (authoritative, in the paper)
  $PY build_cv5_tex.py           > logs/cv5_tex_build.log 2>&1
  # 2) PNG previews (auxiliary)
  $PY build_cv5_table.py         >> logs/cv5_tex_build.log 2>&1 || true
  $PY build_cv5_scaling_figure.py>> logs/cv5_tex_build.log 2>&1 || true
  $PY build_cv5_ablation_table.py>> logs/cv5_tex_build.log 2>&1 || true
  # 3) recompile the paper
  ( cd paper && pdflatex -interaction=nonstopmode genomicrecursiveformer.tex >/tmp/cv5_paperlive_tex.log 2>&1 ) \
    && ok="PDF ok" || ok="PDF FAIL"
  # 4) progress line
  sc=$(ls results_cv5/sc/*/*.json 2>/dev/null|wc -l)
  bsc=$(ls results_cv5/biomor_sc/k*/*/learned_cv.json results_cv5/biomor_sc_token/*/learned_cv.json 2>/dev/null|wc -l)
  mo=$(ls results_cv5/mo/*/*_cv.json 2>/dev/null|wc -l)
  bmo=$(ls results_cv5/biomor_mo/k*/pnet/*/learned_cv.json results_cv5/biomor_mo_token/pnet/*/learned_cv.json 2>/dev/null|wc -l)
  scal=$(ls results_cv5/scaling_sc/*/*.json results_cv5/scaling_sc_biomor/*/*/learned_cv.json results_cv5/scaling_mo/*/*_cv.json results_cv5/scaling_mo_biomor/*/pnet/*/learned_cv.json 2>/dev/null|wc -l)
  abl=$(find results_cv5/ablation -name '*.json' 2>/dev/null|wc -l)
  base=$(ls results_cv5/baselines/*/*_cv.json 2>/dev/null|wc -l)
  left=$(squeue -u tirtho -h -o '%j' -t RUNNING,PENDING | grep cv5 | grep -v cv5-paperlive | wc -l)
  echo "[$(date +%m-%d_%H:%M)] $ok | sc=$sc/64 bsc=$bsc/32 mo=$mo/40 bmo=$bmo/12 scal=$scal/165 abl=$abl/159 base=$base/33 | jobs_left=$left" >> "$LOG"
  sleep 180
done
