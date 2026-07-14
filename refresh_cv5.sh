#!/usr/bin/env bash
# Regenerate the 5-fold-CV ladder table + scaling figure from results_cv5/, and
# print progress counts. Safe to run any time; renders partial ('run…') cells.
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"
$PY build_cv5_tex.py                 # native LaTeX table fragments for the paper (authoritative)
$PY build_cv5_table.py               # standalone PNG previews (auxiliary)
$PY build_cv5_scaling_figure.py
$PY build_cv5_ablation_table.py
# real-time paper: re-embed the freshly rendered CV5 figures (symlinked into paper/figs)
if command -v pdflatex >/dev/null 2>&1; then
  ( cd paper && pdflatex -interaction=nonstopmode genomicrecursiveformer.tex >/tmp/cv5_paper.log 2>&1 ) \
    && echo "  [paper] recompiled genomicrecursiveformer.pdf" || echo "  [paper] compile skipped/failed (see /tmp/cv5_paper.log)"
fi
echo "---- results_cv5 progress ----"
echo "  sc(gen)     : $(ls results_cv5/sc/*/*.json 2>/dev/null | wc -l) / 64"
echo "  biomor_sc   : $(ls results_cv5/biomor_sc/*/*/learned_cv.json results_cv5/biomor_sc_token/*/learned_cv.json 2>/dev/null | wc -l) / 32"
echo "  mo(gen)     : $(ls results_cv5/mo/*/*_cv.json 2>/dev/null | wc -l) / 40"
echo "  biomor_mo   : $(ls results_cv5/biomor_mo/*/pnet/*/learned_cv.json results_cv5/biomor_mo_token/pnet/*/learned_cv.json 2>/dev/null | wc -l) / 12"
echo "  scale_sc    : $(ls results_cv5/scaling_sc/*/*.json 2>/dev/null | wc -l) / 80"
echo "  scale_sc_bio: $(ls results_cv5/scaling_sc_biomor/*/*/learned_cv.json 2>/dev/null | wc -l) / 40"
echo "  scale_mo    : $(ls results_cv5/scaling_mo/*/*_cv.json 2>/dev/null | wc -l) / 30"
echo "  scale_mo_bio: $(ls results_cv5/scaling_mo_biomor/*/pnet/*/learned_cv.json 2>/dev/null | wc -l) / 15"
echo "  ablation    : $(find results_cv5/ablation -name '*_cv.json' -o -name 'baron.json' -o -name '*.json' 2>/dev/null | grep -cE 'ablation/.*\.json') / 159"
echo "  queue       : $(squeue -u tirtho -h -o '%T' -t RUNNING,PENDING | grep -c . ) jobs; running=$(squeue -u tirtho -h -t RUNNING -o '%j' | grep -c cv5)"
