<div align="center">

# SMART / bioMoR — Biology-guided Adaptive Recursive Transformer for Transcriptomic Classification

**Koushik Howlader¹ · Tirtho Roy¹ · Md Tauhidul Islam² · Wei Le¹**

¹ Iowa State University   ² Stanford University

*Reference implementation and full reproduction harness for the paper in `paper/` (`main.tex` + `supplementary.tex`). Manuscript under review — not yet accepted.*

</div>

---

## What this is

**bioMoR** is a biology-guided adaptive recursive transformer for transcriptomic
classification. It compresses the input into a small set of interpretable marker (single-cell)
or Reactome pathway (multi-omics) tokens, applies **one weight-shared block recursively**, and
combines data-driven and biology-prior scores to allocate deeper computation to informative
tokens. A *learned* low-rank interaction graph is injected at **both** the embedding and the
router (zero-init graph-conv) — the paper's main result. Everything is evaluated under a unified
**5-fold cross-validation** protocol (seed 42) across 8 single-cell suites and multi-omics /
P-NET cohorts.

This repository reproduces **every table and figure in the paper** from the committed results in
`results/`, and documents how to regenerate those results from scratch on a GPU cluster.

## Repository layout

```
.
├── recursive_marker_transformer/   # the model + training package (source of truth)
├── genomic_dataloader/ genomap/ bio_networks/   # import-time training dependencies
├── data/               # datasets: singlecell/ + multi-omics cohorts (gitignored, see Data)
├── results/            # committed results the paper is built from
│   ├── cv5/            #   5-fold-CV JSON — feeds almost every table/figure
│   ├── repro/          #   PATH-protocol (10-fold) JSON — supplementary pos-F1 table
│   └── depth/          #   per-pathway depth panels — figure 2
├── scripts/            # table/figure builders + data-gen entry points + refresh_cv5.sh
├── slurm/              # SLURM jobs that (re)produce results/ from scratch
├── paper/              # main.tex, supplementary.tex, cv5_*.tex fragments, figs/, refs.bib
└── archive/            # superseded code / results / notes (gitignored; not needed to reproduce)
```

## Setup

```bash
# Python 3.11. On the GPU box install the CUDA build of torch first, then the rest.
python3.11 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

The developed-against environment is Python 3.11 + `torch 2.10.0+cu128`. All commands below use
that interpreter; export it once for convenience:

```bash
export PY=/work/mech-ai-scratch/tirtho/.venv/bin/python   # or ./.venv/bin/python
```

Run every command **from the repository root**.

## Data

`data/` is gitignored (large). It must contain:

- `data/singlecell/{baron,lung,muraro,oesophagus,segerstolpe,spleen,tcell,xin}/` — the 8
  single-cell suites (+ `manifest.{csv,json}`), materialized via `genomic_dataloader`.
- `data/{prostate,blca,stad,pan_meta_pri,pan_meta_pri_3modal}/` — multi-omics / P-NET cohorts,
  each with `adjacency_matrix.csv` (the provided Reactome pathway graph), `filtered_pathways.csv`,
  and the mutation / CNV / expression / label CSVs.

## Reproduce the paper from the committed results (no GPU needed)

The build scripts only read JSON from `results/` — they run in seconds on CPU. One command
regenerates **all** table fragments + figures and recompiles both PDFs:

```bash
bash scripts/refresh_cv5.sh
```

Or run each piece individually. **Every table/figure in the paper maps to exactly one command:**

| Paper artifact (`paper/…`)            | Regenerate with                         | Reads from `results/` |
|---------------------------------------|-----------------------------------------|-----------------------|
| `cv5_main_table.tex` — Table 2 (efficiency ladder) | `$PY scripts/build_cv5_tex.py`          | `cv5/{sc,mo,biomor_canonical,biomor_ladder,inject_mo,biomor_ladder_mo}` |
| `cv5_ablation_table.tex` — Table 4 (component ablations) | `$PY scripts/build_cv5_tex.py`          | `cv5/ablation` |
| `cv5_baselines_table.tex` — Table 5 (vs classical) | `$PY scripts/build_cv5_tex.py`          | `cv5/baselines` |
| `cv5_scaling_table.tex` — supp. scaling | `$PY scripts/build_cv5_tex.py`          | `cv5/scaling_*` |
| `cv5_injection_table.tex` — **Table 3** (where should biology enter) | `$PY scripts/build_injection_table.py`  | `cv5/{biomor_canonical,inject_mo}` |
| `cv5_posf1_table.tex` — supp. positive-class F1 | `$PY scripts/build_posf1_table.py`      | `repro/ladder` |
| `figs/biorouter_bars.pdf` — Fig. bio-router ablation | `$PY scripts/make_biorouter_bars.py`    | `cv5/biorouter_ablation` |
| `figs/baron_loss.pdf`, `figs/baron_val_f1.pdf` — Fig. training dynamics | `$PY scripts/make_baron_epoch_figs.py`  | `cv5/curves` |
| `figs/fig2_depth.pdf` — Fig. 2 depth panels | `$PY scripts/make_fig2_depth.py`        | `depth` |
| `figs/pareto_efficiency.pdf` — Fig. accuracy–compute Pareto | `$PY scripts/pareto_prototype.py`       | `cv5/` (ladder, analytic FLOPs) |
| `figs/overview.pdf` — Fig. 1 schematic | *static asset* (hand-made, no generator) | — |

`build_cv5_tex.py` writes four fragments at once (`cv5_{main,scaling,ablation,baselines}_table.tex`);
`build_injection_table.py` and `build_posf1_table.py` write one each. Missing/partial cells render
as `run…` placeholders, so the paper always compiles.

## Build the paper

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex          # + bibtex main; pdflatex ×2 for refs
pdflatex -interaction=nonstopmode supplementary.tex
```
(`scripts/refresh_cv5.sh` already recompiles both after regenerating the fragments.)

## Reproduce the results from scratch (GPU cluster)

Retraining regenerates the JSON in `results/`. Each SLURM job writes into the subtree its
table/figure reads. Submit from the repo root; `*_nova.sbatch` twins are the same command on the
`nova` account. Training entry points live in `recursive_marker_transformer/` (invoked as
`python -m recursive_marker_transformer.<module>`) and the dispatchers in `scripts/`.

| Result subtree (`results/…`)          | Produced by (SLURM)                                   |
|---------------------------------------|-------------------------------------------------------|
| `cv5/sc`, `cv5/mo`                     | `slurm/run_cv5_sc.sbatch`, `slurm/run_cv5_mo.sbatch` (+ `run_cv5_panmeta_fix*`) |
| `cv5/biomor_canonical{,_mo}`           | `slurm/run_canonical_biomor_{sc,mo}.sbatch`           |
| `cv5/biomor_ladder{,_mo}`              | `slurm/run_biomorboth_ladder_{sc,mo,mo_pancan}.sbatch`, `slurm/run_cv5_tokenk.sbatch` |
| `cv5/inject_mo` (Table 3)              | `slurm/run_injection_{ablation_sc,mo,pancan,3m}.sbatch`, `slurm/run_3m_cv5_appletoapple_*.sbatch` |
| `cv5/ablation` (Table 4)               | `slurm/run_cv5_ablation.sbatch` → `scripts/ablate_cv5.py --index N` (array) |
| `cv5/baselines` (Table 5)              | `slurm/run_cv5_baselines.sbatch`, `slurm/run_baselines_newmo.sbatch` |
| `cv5/scaling_*` (supp. scaling)        | `slurm/run_cv5_scale_{sc_gen,sc_biomor,mo_gen,mo_biomor}.sbatch` |
| `cv5/biorouter_ablation` (fig)         | `slurm/run_biorouter_{ablation,prostate}.sbatch`      |
| `cv5/curves` (baron dynamics fig)      | `slurm/run_baron_cost.sbatch` → `scripts/make_baron_cost.py` |
| `depth` (fig 2)                        | `slurm/run_prostate_panels.sbatch` → `scripts/prostate_depth_panels.py` |
| `repro/ladder` (supp. pos-F1)          | `slurm/run_ladder_posf1.sbatch`, `slurm/run_repro_all.sbatch` → `scripts/reproduce_path.py` |

After the relevant jobs land, rerun `bash scripts/refresh_cv5.sh` to rebuild the paper.

## Citation

> **Status:** manuscript **under review** — not yet accepted or published.

```bibtex
@unpublished{howlader2026smart,
  title  = {Biology-guided Adaptive Recursive Transformer for Transcriptomic Classification},
  author = {Howlader, Koushik and Roy, Tirtho and Islam, Md Tauhidul and Le, Wei},
  note   = {Manuscript under review},
  year   = {2026}
}
```

## License

See `LICENSE`. Copyright © 2026 the authors. All rights reserved.
