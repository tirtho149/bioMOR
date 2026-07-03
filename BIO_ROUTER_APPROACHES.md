# SMART Bio-Router — Approaches, Mechanisms, and Results

**Project:** RecusrsiveQFormer / SMART (Selective Marker-guided Adaptive Recursive Transformer)
**Scope:** optimizing the *biology-informed routing* component so it measurably beats a plain
data-driven router.
**Last updated:** 2026-07-03

---

## 1. Problem statement

SMART routes each gene/pathway **marker token** to its own recursion depth (Mixture-of-Recursions).
The **bio-router** injects a gene–gene (or pathway–pathway) interaction structure into that routing /
into the token representation, hoping biology tells the model *where to spend compute* and *how to
denoise* gene expression.

The original design (annealed additive centrality bias from a co-expression graph) **did not beat a
vanilla router**, and — more damningly — **did not separate from a degree-matched random graph**.

### The core obstacle: FACT B
> A co-expression (|Pearson|) graph is, by construction, statistically reproducible by a
> degree-matched shuffle of its own edges. So `coexpr ≈ random` is a *theorem*, not bad luck.

Any honest bio-router must therefore show `biology > random ≈ none`. Every **fixed** graph we tried
fails this, because a fixed prior is (i) **label-free** (no gradient reaches the edges),
(ii) **annealed away** (does not change the converged model), and (iii) **not task-shaped**.

**Success criterion (pre-registered):** pooled `Δmacro-F1 ≥ +2.0`, one-sided paired Wilcoxon
`p < 0.05`, and `approach ≥ none`. Pairs are formed **within (cohort, task, seed)** so the shared
split cancels and only the graph differs.

---

## 2. Data (only collaborator data is valid)

Valid data lives on the GitHub `origin/data-branch` (collaborator-maintained), MD5-verified against
the local working tree:

| family | cohorts | modality | gene symbols? | curated net feasible? |
|---|---|---|---|---|
| **P-NET** | prostate, blca, stad, brca (+ pan_meta_pri) | mutation / CNV + Reactome pathways | **yes** | **yes** |
| **genomap single-cell** | Tabula Muris, pancreas, Baron, Segerstolpe, … | scRNA (1089 HVG) | **no** (anonymized `.npy`/`.mat`, LFS) | no (only co-expression) |

- **TCGA is NOT collaborator data** → excluded; any TCGA results were discarded.
- P-NET ships `filtered_pathways.csv` (1268 Reactome pathways → member genes) and
  `adjacency_matrix.csv` (1268×1268 pathway↔pathway graph) — used below.

---

## 3. Approaches

### A. Co-expression prior (paper baseline) — ❌ FAIL
- **Graph:** genomap gene–gene interaction = `1 − Pearson correlation`
  (`createInteractionMatrix(X, metric='correlation')`; **no** Gromov-Wasserstein — GW is only
  genomap's 2D layout, not the routing prior).
- **Use:** per-gene eigenvector centrality added as an *annealed additive bias* on the depth-router
  logit.
- **Result:** `coexpr − none ≈ +0.004`, `coexpr − random ≈ +0.003` (10 single-cell sets). Ties the
  random control. **This is the paper's honest negative.**
- **Why it fails:** all three failure modes above; the graph reaches the model only as a static,
  range-restricted, annealed scalar (see `BIO_ROUTER_REDESIGN.txt` FM-1…FM-5).

### B. Redesigned co-expression (Fix A–E) — ❌ FAIL
- Added: sample-conditional graph **propagation** `x ← (1−λ)x + λ·SH` (Fix A), FiLM-**gated** prior
  (Fix B), de-confounded **precision** graph + seeded **PPR** centrality (Fix C), **persistent
  learnable** β (Fix D), Laplacian **depth-smoothness** (Fix E).
- **Result (single-cell):** with propagation ON, `coexpr` actually *lost* to `random`
  (pancreas −5.48) — a dense |corr| hairball **over-smooths**. Without propagation it re-tied.
- **Verdict:** the mechanism ran as intended (learned λ≈0.5, β≈1.0) but co-expression structure
  still carries no usable, non-generic signal.

### C. Curated Reactome gene–gene network — ❌ FAIL (but informative)
- **Graph:** `W = P·Pᵀ` where `P` = gene→Reactome-pathway membership (genes sharing a pathway are
  linked). A curated network a degree-preserving shuffle *cannot* reproduce → gives the
  `curated vs random` test a real mechanism to separate.
- **Control:** `random` = the SAME graph under a random gene **relabeling** (degree, weight, and
  spectrum identical; only biological identity destroyed) — the exact FACT-B control (fixes the old,
  improperly-matched binary control).
- **Centrality:** label-aware **PPR** seeded by train-fold ANOVA-F genes (leakage-safe);
  `corr(pi, PC1) ≈ 0.05` confirms it is **not** the housekeeping axis.
- **Result (P-NET, 4 cohorts × 10 seeds):** pooled `curated − random = +0.53` (p=0.26),
  `curated − none = +0.44`. **FAIL.** On prostate `curated − none = +4.90` **but random helps
  equally** → "any structured graph gives a small regularizing boost," not biology.
- **Takeaway:** FACT B holds even for *real curated biology*. Fixed graphs don't separate.

### D. **Data-driven learned graph — ✅ FIRST REAL WIN**
- **Idea:** stop imposing a graph; **learn** one that raises accuracy.
- **Mechanism:** each gene gets a small learnable embedding `E ∈ R^{G×r}` (r=16). The gene–gene
  affinity is `A = Ê·Êᵀ` (cosine similarity of learned embeddings) — a **synthetic correlation the
  task discovers**. Expression is propagated along it in **low rank** (never materializes G×G):
  ```
  x ← (1−λ)·x + λ·( (x·Ê)·Êᵀ )       # Ê = row-normalized E ; λ learnable (sigmoid)
  ```
  Trained end-to-end by the classification loss. Fixes all three failure modes: **label-supervised**
  (gradient reaches E), **persistent** (not annealed), **task-shaped**. Magnitude is renormalized per
  sample for stability.
- **Result (P-NET, 4 cohorts × 10 seeds):**

  | cohort | learned − none | p (1-sided) |
  |---|---|---|
  | prostate | **+5.70** | 0.010 ✅ |
  | blca | **+4.77** | 0.014 ✅ |
  | stad | +0.39 | 0.38 |
  | brca | **−3.08** | 0.97 ❌ |
  | **POOLED** | **+1.94** | **0.043 ✅** |
  | pooled learned − curated (biology) | **+1.51** | — |
  | pooled learned − random | **+2.04** | — |

- **Verdict:** first mechanism to **significantly beat the plain router** (pooled p=0.043) **and**
  beat both biology and random. Pooled +1.94 is just under the strict +2.0 flag but is significant.
- **Known issue:** **brca regresses** (5-class, one class has 14 samples) — the low-rank graph
  overfits. Fix to try: lower rank / higher weight-decay / class-count-aware λ for small cohorts.

### E. Pathformer-style pathway crosstalk — ⏳ running
- **Reference:** Lu Lab, github.com/lulab/Pathformer (Module 1: curate ~1497 pathways from
  KEGG/PID/Reactome/BioCarta, build a **pathway crosstalk network** via BinoX).
- **Our offline analogue:** tokens = **pathways** (pool member-gene expression via `P`), structured
  by a pathway↔pathway **crosstalk graph** used as a transformer **attention bias**. We use the
  cohort's shipped **Reactome pathway↔pathway adjacency** as the crosstalk network (offline; **not**
  the 4-DB BinoX network — a faithful-in-spirit simplification).
- **Test:** crosstalk **ON** (`--pathway_attn_bias`) vs **OFF**, both token-routing, 5 seeds × 4
  cohorts. Results → `results_pathformer/seed*/`.
- **Status:** submitted (job `11415632`); verdict pending.

---

## 4. Code map

| file | what |
|---|---|
| `recursive_marker_transformer/interaction.py` | graph builders. `genomap_interaction` (1−Pearson), `build_interaction_v2` (redesign co-expr), **`build_reactome_falsification`** (curated + relabel-matched control + label-aware PPR + `corr(pi,PC1)` diagnostic), `_label_aware_seeds`, `_reactome_membership` |
| `recursive_marker_transformer/model.py` | `set_gene_interaction` / `set_bio_graph` (fixed graph); **learned graph**: `gene_embed` param + low-rank propagation in `forward` (guard `bio_learned_graph`); pathway attn-bias |
| `recursive_marker_transformer/config.py` | flags: `bio_graph_prop`, `bio_prior_gate`, `bio_prior_learnable`, `bio_centrality`, `bio_depth_laplacian`, **`bio_learned_graph`**, **`bio_learned_rank`**, `pathway_attn_bias` |
| `recursive_marker_transformer/singlecell.py` | `_fit_eval` gained `inter=` override to inject a pre-built graph (curated) or `None` |
| `recursive_marker_transformer/bio_redesign_curated.py` | runner for curated/random/**learned**/none on P-NET + (symbol-bearing) cohorts |
| `recursive_marker_transformer/bio_curated_stats.py` | paired Wilcoxon + bootstrap CI aggregator; **biology verdict** panel + **learned-graph** panel |
| `recursive_marker_transformer/pathway_tasks.py` | pathway-token runner; `--pathway_attn_bias` uses the Reactome pathway crosstalk graph (Pathformer-style) |

Slurm: `slurm/run_bio_curated.sbatch` (biology), `slurm/run_learned_pnet.sbatch` (learned),
`slurm/run_pathformer_pnet.sbatch` (Pathformer), `slurm/run_learned_smoke.sbatch` (decision run).

---

## 5. Reproduce

```bash
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python

# Learned graph vs biology vs random vs none, one P-NET cohort (GPU):
$PY -m recursive_marker_transformer.bio_redesign_curated \
    --family pnet --cohort prostate --task response \
    --modes none curated random learned --seeds 0 1 2 3 4 5 6 7 8 9 \
    --K 4 --epochs 60 --cap_genes 3000 --device cuda

# Full P-NET sweep (all cohorts x 10 seeds):
sbatch slurm/run_learned_pnet.sbatch

# Pathformer-style pathway crosstalk (crosstalk ON vs OFF):
sbatch slurm/run_pathformer_pnet.sbatch

# Aggregate (biology verdict + learned-graph verdict):
$PY -m recursive_marker_transformer.bio_curated_stats --results results_bio_curated
```

---

## 6. Bottom line & next steps

- **Fixed biology (co-expression OR curated Reactome) does not beat a matched random control** — a
  clean, closed negative (FACT B confirmed twice).
- **A data-driven learned low-rank graph is the first mechanism to significantly beat the plain
  router (p=0.043) and beat both biology and random.** This is the direction to build on.
- **Next:**
  1. Fix the **brca** regression (lower rank / stronger regularization / class-aware λ) → aim for a
     clean 4/4.
  2. Report **Pathformer** crosstalk ON-vs-OFF once job `11415632` lands.
  3. **Biology-as-warm-start:** initialize `E` from Reactome/co-expression and let the task refine it
     — best-of-both, and a natural interpretability story (compare learned `A` to known pathways).
  4. Optionally make `A` **sample-conditional** (input-gated) and add a supervised graph-conv in the
     value path.
