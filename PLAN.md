# SMART — Project & Paper Plan

> **Rename:** the model/paper is now **SMART** (Selective Marker-guided Adaptive
> Recursive Transformer). All paper text, the title, and the abstract use "SMART";
> "GenomicRecursiveFormer" below is the former name, kept only for history.

## Goal
Deliver a **fully functional, reproducible pipeline** that (1) trains the
SMART model, (2) runs the ablation suite, (3) emits a conference
`.tex` paper with **every number injected from real experiment results**, and
(4) compiles it to PDF — all from **one shell script** (`run_all.sh`).

The paper is written in confident first-person plural ("we") at the level of a
strong 4th-year PhD student, using a conference LaTeX class, with **25+ verified,
highly connected citations**.

## The idea (punch line)
**Parameter efficiency as an architectural property, not post-hoc compression.**
The model learns which genes are *markers* worth dedicated computation, compresses
everything else into marker-anchored cluster tokens (O(N²)→O(M²) attention), and
applies a *single* transformer block recursively K times (weight sharing), re-scoring
markers after each pass (recursive marker refinement).

## Authors (from PATH.pdf + Tirtho Roy inserted 2nd)
1. Koushik Howlader — Iowa State University
2. **Tirtho Roy** — Iowa State University  *(2nd author, as requested)*
3. Md Tauhidul Islam — Stanford University
4. Wei Le — Iowa State University

## Data / target
Bulk pan-cancer TCGA via `genomic_dataloader` (4 cohorts), primary head
`cancer_type`. Architecture is data-agnostic → single-cell cell-type labels later.

## Repository layout (new/changed)
```
recursive_marker_transformer/
  config.py        RMTConfig dataclass (all knobs)
  data.py          genomic_dataloader wrapper; correct raw-variance HVG; label remap
  embedding.py     gene-identity + value embedding
  marker.py        MarkerHead, differentiable top-k, fixed random panel, aggregation, refine
  recursion.py     SharedTransformerBlock + RecursiveStack (shared/independent, adaptive depth)
  model.py         RecursiveMarkerTransformer (+ param counters)
  losses.py        task + marker + diversity + compression
  train.py         run(cfg) -> results dict; per-class report for EVERY head
  experiments.py   runs SUITE + exact param-efficiency table -> results/*.json
  make_paper.py    reads results/*.json -> paper/genomicrecursiveformer.tex + refs.bib
  bio_enrichment.py Reactome enrichment of learned markers (Ablation 9)
  ablate.py        ad-hoc ablation presets
run_all.sh         one command: experiments -> paper -> PDF
PLAN.md            this file
aaai_template/     conference LaTeX style files (.sty/.bst, fixbib.sty)
results/           generated JSON (numbers feed the paper)
paper/             generated .tex/.bib + compiled PDF
```

## Experiment suite (`experiments.py` SUITE)
- `main` — learnable markers + shared recursion (headline)
- `random_markers` — random panel (shows marker learning matters)
- `variance_markers` — variance selection (middle baseline)
- `independent` — no weight sharing (param ablation)
- `no_refine` — recursive refinement off
- `depth1` — no recursion (K=1)
- **param_efficiency** — exact shared-vs-independent param counts across K (no training)

Scale via flags/env: `RMT_EPOCHS`, `RMT_NHVG`, `RMT_DMODEL`, `RMT_NMARKERS`,
`RMT_DEPTH`, `RMT_HEADS`. Quick mode for smoke; full mode for real numbers.

## Paper structure (`make_paper.py` template, tokens filled from JSON)
Abstract · Introduction (contributions) · Related Work (foundation models;
parameter-efficient/recursive transformers; markers/pathways) · Method (5 stages +
objective + complexity) · Experiments (setup; main table; per-class table; param
table; ablation table; biological enrichment) · Discussion/Limitations · Conclusion ·
References (`refs.bib`, conference `.bst`).

Tables auto-built: `main_results_table`, `per_class_table`, `ablation_table`,
`param_table`. No number hand-typed.

## Citations (25+, web-verified)
Vaswani'17, Dehghani'19 (Universal Transformers), Lan'20 (ALBERT), Bae'25 (MoR),
Raposo'24 (MoD), Shazeer'17 (MoE), Graves'16 (ACT), Wang'20 (Linformer),
Choromanski'21 (Performer), Xiong'21 (Nyströmformer), Jang'17 (Gumbel-softmax),
Cui'24 (scGPT), Hao'24 (scFoundation), Theodoris'23 (Geneformer), Yang'22 (scBERT),
Ianevski'22 (scType), Islam&Xing'23 (genomap), Hu'23 (CellMarker 2.0),
Franzén'19 (PanglaoDB), Gillespie'22 (Reactome), Tabula Sapiens'22, Wolf'18 (scanpy),
Regev'17 (HCA), Aran'19 (SingleR), Luecken&Theis'19, Howlader'26 (PATH),
Lopez'18 (scVI), Weinstein'13 (TCGA). = 28.

## `run_all.sh` flow
1. resolve `.venv` python (py3.11 + torch 2.2.2 + numpy<2 on this Intel Mac).
2. `python -m ...experiments` (configurable scale) → `results/`.
3. `python -m ...make_paper` → `paper/`.
4. `pdflatex → bibtex → pdflatex×2` → `paper/genomicrecursiveformer.pdf`.

## Reproducibility notes
- Env: Intel macOS → torch's last wheel is 2.2.2, requires `numpy<2`; default
  `python3` is 3.14 (no torch wheel), so we use a `python3.11` venv.
- CPU-only here; full suite is slow (~10–15 min/run). `run_all.sh` defaults to a
  tractable config and exposes env knobs to scale up on GPU.
- Bug fixes already landed: raw-variance HVG (loaders return z-scored data),
  scaler-floor handling, marker aux-loss on cluster tokens, fixed random panel,
  safe label remap.

## Marker mechanism: BEST approach = cross-attention router (headline)
Empirically determined the best marker selector (M=48, 8ep diagnostic, macro-F1):
- **Cross-attention slot router (ours): 0.966** 🏆  (M marker queries soft-attend
  over all genes; all-gene gradient; **peaked init**; temperature anneal; hard
  arg-max at eval)
- variance heuristic: 0.957 | Concrete: 0.943 | random: 0.903
- naive hard noisy-top-k router: 0.70 | router w/ uniform init: 0.32 (both fail)

**Two decisive ingredients:** (1) soft selection over ALL genes (gradient
everywhere → can discover genes; hard top-k can't explore) and (2) peaked init
(start at random-selection quality, not uniform mush). `marker_mode="router"` is
the headline; `concrete` is a documented alternative; `random`/`variance` are
baselines. Naive router kept in narrative as the informative failure.

## (earlier) Concrete selection
A first run showed **hard top-k marker selection underperforms random** (it can
only re-rank a frozen set, never explore new genes). We replaced it with a
**Concrete / Gumbel-softmax feature-selection layer** (Balın et al. 2019): M
selectors, each a temperature-annealed (10→0.1) distribution over all N genes, so
gradients reach every gene and the model learns which genes are markers. Eval uses
hard arg-max per selector. This is now `marker_mode="concrete"` (the headline
model); `random`/`variance` remain as selection baselines; `aggregate`/`drop`
kept as compression variants. Verified: gradients flow to all genes, annealing
works, 126/128 distinct markers.

## Status
- [x] Model + data + training (verified: ~0.95 acc, 4× param cut)
- [x] All bugs fixed; structural smoke passes
- [x] 25+ citations verified (now 29, incl. Concrete Autoencoder)
- [x] experiments.py, make_paper.py, run_all.sh written
- [x] LaTeX pipeline proven (5-page PDF, all refs resolve, placeholders + concrete prose)
- [x] Concrete selection implemented + verified
- [~] Full pipeline running (6 experiments, epochs=12) → real PDF
- [ ] Confirm learned > random/variance in selection study; ship final PDF
```

## Update — paper polish, single-cell extension, data consolidation

### Paper polish (done)
- Renamed everything to **SMART**; title is "SMART: Selective Marker-guided
  Adaptive Recursive Transformer for Transcriptomic Classification".
- Two **TikZ figures** (system overview with shaded A/B panels + fontawesome icons;
  MoR mechanism = shared block internals + adaptive-depth funnel grid). Overlaps
  fixed (straddling panel tabs, rerouted connector, spaced capacity labels).
- **Per-class (per-cohort) table** named by cohort (BRCA/HNSC/LUNG/THCA); class
  names persisted from `train.py` (`_CANCER_RAW_NAMES`).
- Wide tables wrapped in `\resizebox{\columnwidth}`; gene-id table made a wrapping
  single-column table next to its section.
- **Results sync verified** (0 unresolved tokens); fixed Fig.1 `N=4000→@@NGENES@@`
  (=2000) and τ schedule prose `10→1`.
- **De-AI prose pass**: removed all em/en dashes from the body (kept compound-term
  hyphens), softened formulaic phrasing.

### Data layout — single `data/` home (done)
All data now lives under `data/`:
- `data/tcga/` — TCGA bulk RNA-seq CSVs (UCSC Xena HiSeqV2); `genomic_dataloader`
  `_ROOT` updated to point here (with legacy fallback).
- `data/singlecell/<dataset>/` — converted single-cell CSVs.
Deleted as unnecessary: `capsule-6967747-data.zip` (2.2 GB), redundant capsule
`.mat` files, old `results_v1_old/`, reference clones (`TherapAgent`,
`Recursive-Transformer`, `mixture_of_recursions`, `Q-Learning-Algorithm`),
`__pycache__`, LaTeX build artifacts. Freed ~8.4 GB.

### Single-cell generalization (data verified RNA-seq; runner built; **run pending on GPU**)
- Source: genomap capsule **6967747** (CodeOcean). `tools/convert_capsule_to_csv.py`
  converts its `.mat`/`.csv` to readable CSVs (dynamic, reads the zip directly,
  deterministic): expression.csv.gz + labels.csv + split.csv + manifest.
- Datasets: **tabula_muris** (54,865 cells, 1,089 genes, 55 classes; own split),
  **common_class** (27,499 / 1,089 / 19), **prototype** (90,579 / 752 / 10),
  **pancreas** (14,767 / 1,936 / 15; features = flattened 44×44 genomaps).
- `recursive_marker_transformer/singlecell.py` runs the **headline SMART config**
  (router selection + expert-choice MoR) per dataset → `results_singlecell/*.json`.
- Paper: new subsection **"Generalization to Single-Cell Datasets"** +
  `@@SINGLECELL_TABLE@@` (Dataset / Cells / Genes / Classes / Acc / Macro-F1),
  auto-filled from those JSONs.
- **Lesson learned:** large batch + few epochs badly undertrains (TM 11%, comClass
  22%). Correct settings = smaller batch + higher LR + more epochs+patience
  (diagnostic: comClass val macro-F1 0.057 → ~0.55). Use
  `--epochs 15 --batch_size 256 --lr 1e-3 --patience 5` on CPU, or larger batch on GPU.

### GPU transfer (next step — CPU here is too slow)
This Intel Mac has no usable GPU (GT 755M unsupported; `mps` needs Apple Silicon),
so the single-cell run is ~2.5–3 h on CPU and was **terminated for transfer**.
On a CUDA box it finishes in ~5–15 min. To run there:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # install the +cuXXX torch wheel
python -m recursive_marker_transformer.singlecell --epochs 15 --batch_size 1024 --lr 1e-3
python -m recursive_marker_transformer.make_paper --results results --outdir paper
# then pdflatex → bibtex → pdflatex×2
```
Transfer the repo **without `.venv/`** (recreate from `requirements.txt`).

### Remaining
- [ ] Run `singlecell.py` on a GPU → fill the single-cell table.
- [ ] Sanity-check the genomap-derived datasets (common_class/pancreas may be lower,
      as features are genomap images, not raw gene panels); soften the table prose if so.
- [ ] Address the reviewer asks (see memory `review-action-items`):
      external baselines, 20k-gene scaling, multi-seed ± std, real enrichment.
