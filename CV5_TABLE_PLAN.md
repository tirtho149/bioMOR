# bioMoR Ladder Table — 5-Fold CV Rebuild Plan

**Goal.** Replace the current *single-split × 3-seed* protocol behind `biomor_ladder_table.png`
with a rigorous, unified **5-fold cross-validation** protocol and report **mean ± SD** in
**every cell** of the table. Every cell (14 rows × 13 datasets) is re-run **fresh**.

## 1. Split protocol (identical for ALL rows / datasets / variants)

Decided first, once, and reused everywhere:

- **5-fold CV**, `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
  - Each fold ⇒ **20 % test**, **80 % train** (this *is* 5-fold: the 5 test folds tile the data).
- Within each fold's 80 % train, hold out **10 % of train as validation**
  (`train_test_split(test_size=0.10, random_state=42, stratify=y_train)`).
  ⇒ per fold: **72 % train / 8 % val / 20 % test** of the full dataset.
- **seed = 42** everywhere. Because the fold assignment depends only on `(y, seed)` and the
  data is loaded deterministically, **the 5 folds are byte-identical across every variant** —
  comparisons are paired.
- Fresh training each fold: **max epochs = 100**, **early-stop patience = 15**, best-val checkpoint.
- Per (variant, dataset): **macro-F1 mean ± SD over the 5 folds** → one cell.

Implemented once in `recursive_marker_transformer/cv.py::cv_folds()` and called by all
four training entry points, so the protocol cannot drift between rows.

## 2. Rows → command map (what each cell comes from)

| Table row group | Entry point | Variant flags |
|---|---|---|
| **Vanilla** | SC: `singlecell` · MO: `pathway_tasks` | `--recursion_mode expert --no_share_weights` (`independent`) |
| **Recursive** ×3 | same | `--recursion_mode fixed --recursion_depth {2,3,4}` (`fixed_k2/k3/fixed`) |
| **MoR (general)** ×4 | same | `--recursion_mode expert --recursion_depth {2,3}` , `expert` (`shared`), `--recursion_mode token` |
| **bioMoR ladder** ×2 | SC: `bio_learned_genomap` · MO(Pro/BL/ST): `bio_redesign_curated` | `--modes learned --K {2,3}` |
| **bioMoR + Token** | `bio_learned_genomap` / `bio_redesign_curated` | `--modes learned --K 4 --recursion_mode token` |
| **bioMoR (ours)** | `bio_learned_genomap` / `bio_redesign_curated` | `--modes learned --K 4` |

- **Single-cell (8 cols)**: baron, lung, muraro, oesophagus, segerstolpe, spleen, tcell, xin.
- **Multi-omics core (Pro/BL/ST)**: `pathway_tasks` (general rows) / `bio_redesign_curated` (bioMoR rows),
  cohorts prostate/blca/stad, `--channels mut_cnv --marker_mode pathway --gene_interaction reactome`.
- **Pan-cancer extras (PM/PC)**: `pathway_tasks` tasks `pan_meta_pri` (mut_cnv) and `panmeta_response` (expr),
  one CV run per general variant; bioMoR rows reuse the matching pathway-side variant (as today).

## 3. Output layout (clean, uniform)

All CV results land under a NEW tree so nothing collides with the legacy single-split dirs:

```
results_cv5/
  sc/<variant>/<dataset>.json            # singlecell CV
  biomor_sc/k{2,3,4}/<Dataset>.json      # bio_learned_genomap CV
  biomor_sc_token/<Dataset>.json         # token-choice bioMoR SC
  mo/<variant>/<tag>.json                # pathway_tasks CV (Pro/BL/ST/PM/PC)
  biomor_mo/k{2,3,4}/<cohort>.json       # bio_redesign_curated CV
  biomor_mo_token/<cohort>.json
```

Every JSON carries: `macro_f1: {mean, std}` (percent), `fold_f1: [...]`, `n_folds`, `n_samples`,
`n_classes`, `config`. (mean/std in **percent**, so the table reads them directly.)

## 4. Steps

1. [x] Audit pipeline + confirm split interpretation & inline ±SD.
2. [x] `cv.py` shared fold helper (StratifiedKFold-5, seed 42, val 0.10).
3. [x] Add `--cv_folds` CV mode to all 4 entry points → `results_cv5/` (uniform `cv_macro_f1` percent).
4. [x] CPU smoke: singlecell/bio_learned/bio_redesign write valid JSON; pathway path reached (login-node OOM only). GPU-validated: `baron` = 67.6±2.2.
5. [x] CV sbatch arrays submitted.
6. [~] Monitoring to completion (`./refresh_cv5.sh`).
7. [x] `build_cv5_table.py` — inline `mean±sd` every cell.
8. [~] Auto-regenerate `biomor_ladder_table_cv5.png` as results land.

## 6. Model-SIZE scaling (same split) — user follow-up

Re-ran the width sweep (d_model 96/136/192/272/352 × {Vanilla, MoR-general, bioMoR} × {SC, MO})
under the identical 5-fold CV protocol → `results_cv5/scaling_*`, figure `biomor_scaling_figure_cv5.png`
via `build_cv5_scaling_figure.py`.

## SLURM job IDs (2026-07-13)

Table: SC=11632222 · bioMoR-SC=11632223 · MO=11632224 · bioMoR-MO=11632225
Scaling: SC-gen=11632286 · SC-bio=11632287 · MO-gen=11632288 · MO-bio=11632289
(313 array tasks total, scavenger, `--requeue`.)

## 5. Compute

~14 variant-configs × (8 SC + 5 MO) datasets, each a 5-fold CV job (5 fresh trainings).
Datasets are small (≤ tens of k rows); 100-epoch fold ≈ minutes on a GPU. Run as scavenger
array jobs across many GPUs (`--array ...%N`, N large). Est. a few hours wall-clock.
