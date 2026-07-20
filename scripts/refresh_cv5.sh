#!/usr/bin/env bash
# ============================================================================
# ONE-SHOT paper refresh: regenerate every table fragment + figure the paper
# needs from the existing results/ trees, then recompile the PDFs.
# Reads only JSON (CPU, seconds) -- safe to run any time; partial results render
# as 'run...' placeholder cells. Run from anywhere:  bash scripts/refresh_cv5.sh
# ============================================================================
set -u
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root, relative to this script
PY="$PROJ/../.venv/bin/python"                             # project venv if present, else PATH python
[ -x "$PY" ] || PY="$(command -v python3 || command -v python)"
cd "$PROJ"

echo "== table fragments (authoritative, \\input by the paper) =="
$PY scripts/build_cv5_tex.py          # cv5_{main,baselines,ablation,scaling}_table.tex
$PY scripts/build_injection_table.py  # cv5_injection_table.tex  (Table 3, main contribution)
$PY scripts/build_posf1_table.py      # cv5_posf1_table.tex      (supplementary appendix)
$PY scripts/build_pm_depth_tables.py  # pm_depth_tables.tex      (supplementary per-pathway depth)

echo "== figures =="
$PY scripts/make_biorouter_bars.py   || echo "  [warn] biorouter_bars skipped"
$PY scripts/make_baron_epoch_figs.py || echo "  [warn] baron_curves skipped (needs results/cv5/curves)"
$PY scripts/make_baron_pointdiagram.py || echo "  [warn] baron_pointdiagram skipped (needs results/cv5/curves)"
$PY scripts/make_fig2_depth.py       || echo "  [warn] fig2_depth skipped (needs results/depth)"
$PY scripts/make_pm_depth_figure.py  || echo "  [warn] pm_depth figure skipped (needs results/cv5/pm_routing)"
$PY scripts/pareto_prototype.py      || echo "  [warn] pareto_efficiency skipped"

echo "== compile paper (pdflatex x2 -> bibtex -> pdflatex x2, so refs/citations resolve) =="
if command -v pdflatex >/dev/null 2>&1; then
  for doc in main supplementary; do
    ( cd paper \
      && pdflatex -interaction=nonstopmode "$doc.tex" >/tmp/cv5_${doc}.log 2>&1 \
      && pdflatex -interaction=nonstopmode "$doc.tex" >>/tmp/cv5_${doc}.log 2>&1 \
      && bibtex "$doc"                                 >>/tmp/cv5_${doc}.log 2>&1 \
      && pdflatex -interaction=nonstopmode "$doc.tex" >>/tmp/cv5_${doc}.log 2>&1 \
      && pdflatex -interaction=nonstopmode "$doc.tex" >>/tmp/cv5_${doc}.log 2>&1 ) \
      && echo "  [paper] recompiled $doc.pdf" || echo "  [paper] $doc compile issue (see /tmp/cv5_${doc}.log)"
  done
else
  echo "  [paper] pdflatex not on PATH -- skipped"
fi

echo "---- results/cv5 progress ----"
echo "  sc(gen)     : $(ls results/cv5/sc/*/*.json 2>/dev/null | wc -l) / 64"
echo "  mo(gen)     : $(ls results/cv5/mo/*/*_cv.json 2>/dev/null | wc -l) / 40"
echo "  scale_sc    : $(ls results/cv5/scaling_sc/*/*.json 2>/dev/null | wc -l) / 80"
echo "  scale_mo    : $(ls results/cv5/scaling_mo/*/*_cv.json 2>/dev/null | wc -l) / 30"
echo "  ablation    : $(find results/cv5/ablation -name '*.json' 2>/dev/null | grep -cE 'ablation/.*\.json') / 159"
