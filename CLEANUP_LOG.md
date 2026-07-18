# Repo Cleanup & Reproducibility Log

Goal: archive all unused/legacy code into `archive/`, keep only the files needed to
reproduce the **current** paper (`paper/main.tex` + `paper/supplementary.tex`, the
bioMoR / 5-fold-CV story), and rewrite the top-level `README.md` so every table and
figure in the paper has an exact, runnable reproduction command.

Status legend: ⬜ todo · 🔄 in progress · ✅ done

## Phase 0 — Investigation (map the dependency graph)
- ✅ Map every paper table/figure → build script → result dir → training slurm/entry point
- ✅ Classify every top-level file / dir as KEEP (paper pipeline) vs ARCHIVE (legacy)

### VERIFIED FACTS (2026-07-18)
- Current paper = `paper/main.tex` + `paper/supplementary.tex` (bioMoR / 5-fold CV). README.md is STALE (documents old SMART "EXP 1-11").
- Only THREE result dirs are read by paper build scripts: `results_cv5/` (most), `results_repro/` (posf1 appendix table), `results_depth/` (fig2_depth). ALL other `results_*` dirs are legacy → ARCHIVE.
- Paper build scripts (KEEP): build_cv5_tex.py, build_injection_table.py, build_posf1_table.py, make_biorouter_bars.py, make_baron_epoch_figs.py (+make_baron_cost.py data), make_fig2_depth.py (+prostate_depth_panels.py data), pareto_prototype.py, ablate_cv5.py, reproduce_path.py; orchestrated by refresh_cv5.sh. Aux PNG previews: build_cv5_table.py, build_cv5_scaling_figure.py, build_cv5_ablation_table.py.
- Package `recursive_marker_transformer/` KEEP whole; import-reachable core = cv, singlecell, pathway_tasks, pathway_data, pathway_warmstart, depth_sweep, depth_viz, baselines11, bio_learned_genomap, bio_redesign_curated, bio_network, config, data, embedding, interaction, losses, marker, model, recursion, router, train. (make_paper.py etc are legacy/dead but harmless.)
- Training data root `data/`, plus `genomic_dataloader/`, `genomap/`, `bio_networks/` are training deps → KEEP.
- `figs/overview.pdf` = static hand-made asset (no generator).
- NOTE: build scripts hardcode `ROOT=/work/.../RecusrsiveQFormer` and read `results_cv5/...`; slurm scripts write `--out results_cv5/...`. Any dir RENAME requires path rewrites.

### CAVEATS to handle
- pareto_prototype.py may be stale vs current build_cv5_tex.row_vals (3-arg vs 4-arg) — verify/fix during Phase 4.
- refresh_cv5.sh does NOT call the figure scripts (biorouter/baron/fig2/pareto) — README must list them explicitly.
- posf1 table uses results_repro (10-fold reproduce_path.py), a different protocol than the rest — document as such.

## Phase 1 — Plan & approval
- ✅ Presented KEEP / ARCHIVE lists + target layout; user approved full clean reorg + archive orphan paper fragments

## Phase 2 — Reorganize & archive
- ✅ New top-level layout: recursive_marker_transformer/ genomic_dataloader/ genomap/ bio_networks/ data/ results/{cv5,repro,depth} scripts/ slurm/ paper/ archive/ + README/CLEANUP_LOG/LICENSE/requirements
- ✅ Renamed results_cv5→results/cv5, results_repro→results/repro, results_depth→results/depth
- ✅ Moved 12 paper-build/data scripts + refresh_cv5.sh → scripts/
- ✅ Archived: 29 legacy results_* → archive/old_results/; 82 legacy slurm → archive/old_sbatch/ (kept 61);
  12 dead build/util scripts → archive/legacy_scripts/; 12 plan/notes → archive/notes/; root scratch pngs/pdfs
  → archive/scratch_figs/; aux dirs (lit_pipeline, teaser_pipeline, reviews, docs, assets, tools, new data,
  genomap_data, logs, paper_review, aaai_template) + one-off shells → archive/misc/; orphan paper fragments →
  archive/paper_pre_cv5_2026-07-13/

## Phase 3 — Path rewrites & script fixes
- ✅ Token-migrated results_cv5/repro/depth → results/… across all scripts/ + kept slurm (0 stale)
- ✅ Prefixed moved-script invocations in slurm with scripts/ (24 refs); recreated logs/
- ✅ Fixed __file__-anchored ROOT (build_injection_table, make_baron_cost, make_baron_epoch_figs,
  make_biorouter_bars now .parent.parent / dirname(dirname))
- ✅ Added repo-root sys.path bootstrap to package-importers moved into scripts/ (reproduce_path,
  prostate_depth_panels, make_baron_cost)
- ✅ FIXED pareto_prototype.py: 3-arg→4-arg row_vals(kind,variant,K,mode) + output path + docstring (was broken)
- ✅ Rewrote scripts/refresh_cv5.sh into the single one-shot: 3 table builders + 4 figure scripts + compile both PDFs
- ✅ Dropped the 3 aux PNG-preview builders + biorouter_ablation_table.tex byproduct (not used by paper) → clean root

## Phase 4 — Verify (DONE, all green)
- ✅ `grep results_(cv5|repro|depth)` in scripts/ slurm/ → 0 stale
- ✅ All 6 table fragments regenerate rc=0 with real numbers (not placeholders)
- ✅ All 4 figure scripts regenerate rc=0 (incl. previously-broken pareto)
- ✅ `bash scripts/refresh_cv5.sh` → main.pdf + supplementary.pdf recompiled rc=0
- ✅ No kept file references an archived dir/script; root + paper/ clean
- ⚠️ NOT committed — left for user review (git: many D = archived tracked files removed from repo, as intended)

## FINAL LAYOUT
recursive_marker_transformer/ · genomic_dataloader/ · genomap/ · bio_networks/ · data/ (3.4G) ·
results/{cv5,repro,depth} (14M) · scripts/ (11 py + refresh_cv5.sh) · slurm/ (61 sbatch) ·
paper/ (9 tex + figs + refs) · archive/ (27G, gitignored) · README.md · CLEANUP_LOG.md · LICENSE · requirements.txt

---

## Dependency map (current paper)

### main.tex artifacts
| Artifact | Build script | Reads | Training slurm |
|---|---|---|---|
| figs/overview.pdf | recursive_marker_transformer/make_paper.py | (schematic) | — |
| figs/fig2_depth.pdf | make_fig2_depth.py | results_depth/ | (tbd) |
| cv5_main_table.tex | build_cv5_tex.py | results_cv5/{sc,mo,biomor_canonical,biomor_ladder,inject_mo,biomor_ladder_mo} | (tbd) |
| cv5_injection_table.tex | build_injection_table.py | results_cv5/inject_* | (tbd) |
| cv5_ablation_table.tex | build_cv5_tex.py | results_cv5/ablation | (tbd) |
| cv5_baselines_table.tex | build_cv5_tex.py | results_cv5/ | (tbd) |
| figs/biorouter_bars.pdf | make_biorouter_bars.py | results_cv5/ | (tbd) |
| figs/baron_loss.pdf, baron_val_f1.pdf | make_baron_epoch_figs.py | results_cv5/ | (tbd) |
| figs/pareto_efficiency.pdf | pareto_prototype.py | (tbd) | (tbd) |

### supplementary.tex artifacts
| Artifact | Build script | Reads | Training slurm |
|---|---|---|---|
| cv5_posf1_table.tex | build_posf1_table.py | results_repro/ | (tbd) |
| cv5_scaling_table.tex | build_cv5_tex.py | results_cv5/scaling_* | (tbd) |

Orchestration: `refresh_cv5.sh` runs all table/figure builders then recompiles the paper.

(tbd cells filled in Phase 0 by investigation.)
