# Data Preprocessing & Filtering — Pan-Cancer Survival Baselines

This document describes how the multi-omics TCGA data is loaded, filtered,
imputed, split, and normalized for the survival (Cox) baseline suite. Every
baseline consumes the **same** preprocessed data through a single shared
contract (`pan_survival_data.py`) so the comparison is apples-to-apples.

---

## 1. Data sources

The cohort is TCGA pan-cancer, derived from UCSC Xena PANCAN files
(`data_tcga/pancan_raw/`):

| Modality | Raw source file | Type |
|----------|-----------------|------|
| Copy-number (CNV) | `Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes.gz` | GISTIC2 thresholded, integer −2…+2 |
| Mutation | `mc3.v0.2.8.PUBLIC.nonsilentGene.xena.gz` | Non-silent gene mutation, binary 0/1 |
| Gene expression | `EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz` | Batch-adjusted RNA-seq V2, log2 |
| Survival labels | `Survival_SupplementalTable_S1_20171025_xena_sp` | OS / OS.time (+ DSS/PFI/DFI) |

Two prepared cohorts are used:

| Cohort dir | Modalities | Patients | Genes (shared) | Events |
|------------|-----------|----------|----------------|--------|
| `pan_survival_cox`  | CNV + mutation | **8755** | **23384** | 2576 (29.4%) |
| `pan_survival_3mod` | CNV + mutation + expression | **8454** | **17940** | 2404 (28.4%) |

The 3-modal cohort is selected via the environment flag `SURV_3MODAL=1`; the
loader otherwise defaults to the 2-modal cohort.

---

## 2. Loading & alignment pipeline (`load_modalities`)

The shared loader performs the following deterministic steps:

1. **Read each modality** as a patients × genes matrix (patients indexed by
   TCGA barcode).
2. **Gene filtering — intersection across modalities.** Only genes present in
   *all* active modalities are kept, preserving the CNV column order for
   reproducibility:
   ```python
   common_genes = set.intersection(*[set(frames[n].columns) for n in MODALITY_NAMES])
   genes = [g for g in frames["cnv"].columns if g in common_genes]
   ```
   This drops modality-specific genes so every modality is defined on an
   identical gene axis (2-modal → 23384 genes; adding expression → 17940).
3. **Survival-label filtering.** From the survival table keep the chosen
   endpoint (default `OS`) and its time `OS.time`:
   - drop rows with missing event or time,
   - **drop non-positive follow-up** (`OS.time > 0`) — zero/negative durations
     are uninformative for the Cox risk set.
4. **Patient filtering — intersection.** Keep only patients that have *every*
   modality **and** a valid survival label; patients are then sorted for a
   stable, reproducible order.
5. **Missing-value imputation** (see §3).
6. Return `(mods, time, event, patients, genes)` with all matrices aligned
   row-for-row to `patients`.

---

## 3. Missing-value imputation

CNV and mutation matrices contain **no** missing values. Gene expression
contains **~1.34M NaNs** (≈921 patients × up to 3264 genes).

Left untreated, a single NaN poisons downstream models: `StandardScaler.fit`
does not ignore NaN (its mean/std become NaN), and once a NaN enters a graph
convolution (MOGONET) or attention softmax (Pathformer) it propagates to
*every* output, collapsing the C-index to 0.

**Imputation strategy — per-gene cohort mean**, applied once in the loader so
all baselines receive identical clean data:

```python
for name, arr in mods.items():
    if np.isnan(arr).any():
        col_mean = np.nanmean(arr, axis=0)                     # gene mean, ignoring NaN
        col_mean = np.where(np.isnan(col_mean), 0.0, col_mean) # all-NaN gene -> 0
        idx = np.where(np.isnan(arr))
        arr[idx] = np.take(col_mean, idx[1])                   # NaN -> that gene's mean
```

Rationale: mean imputation is the standard treatment for missing expression
values and is *neutral* (it does not bias a gene up or down). Filling with 0
would be wrong — in log2 RNA-seq space 0 means "not expressed," which would
conflate *missing* with *silent*.

*Caveat:* the mean is computed over the full cohort (not per-fold), a small and
conventional form of imputation leakage that uses only a feature's marginal
average — never the labels — and is applied identically to every model.

---

## 4. Cross-validation splits (`make_folds`)

Identical folds for every baseline, so no model gets an easier split:

- **5-fold `StratifiedKFold`** stratified on the **event indicator**, so each
  fold has a comparable censoring rate.
- **Seed 42**, shuffle on.
- Within each training portion, a **validation set = 10%** is carved out, again
  stratified on the event indicator.
- Net split ≈ **72% train / 8% validation / 20% test** per fold.
- Test folds: ~1751 patients each (2-modal) / ~1691 each (3-modal), ~515 / ~481
  events respectively.

Validation is used only for **early stopping** (model selection on validation
C-index); test is untouched until final scoring.

---

## 5. Per-modality normalization (fit on training data only)

Normalization statistics are estimated on the **training fold only** and then
applied to validation and test, to avoid leakage.

| Modality | Treatment | Rationale |
|----------|-----------|-----------|
| CNV | **Z-score** (`StandardScaler`, train-fit) | continuous GISTIC scores |
| Expression | **Z-score** (`StandardScaler`, train-fit) | continuous log2 RNA-seq |
| Mutation | **left binary** (no scaling) | 0/1 indicator; z-scoring would distort a Bernoulli variable |

Generic rule used everywhere: *every modality except `mutation` is z-scored;
`mutation` is passed through as binary.* The single exception is Pathformer,
which stacks all modalities into one tensor once (outside the fold loop) and
therefore standardizes continuous modalities with global statistics.

---

## 6. Pathway-based filtering (pathway-aware models)

`filtered_pathways.csv` provides **1268 curated pathways** (columns
`Pathway_ID, Pathway_Name, Genes`). Three baselines restrict/organize the gene
space by pathway membership:

- **P-NET** — builds a sparse binary gene→pathway **mask** `(G × P)`; only
  gene→pathway edges that exist in the curated set are learnable.
- **Pathformer** — selects the subset of genes that appear in *any* pathway
  (`gene_select`), then builds the same gene→pathway mask plus a
  pathway-crosstalk adjacency from `adjacency_matrix.csv`.
- **PathCNN** — groups genes by pathway and requires **≥5 genes per pathway**;
  each pathway is compressed by **PCA (2 principal components)** fit on the
  training fold, producing a per-pathway "image" (with `np.nan_to_num` guards).

CNN_ei, CNN_li, MOGONET and MOGAT use the **full** shared gene set (no pathway
restriction).

---

## 7. Patient-similarity graphs (graph models)

Graph baselines build a patient×patient graph from the (normalized) features:

- **MOGONET** — cosine **k-NN graph** (k = 10), symmetrically normalized; one
  graph per modality/view.
- **MOGAT** — per-modality similarity via matrix algebra (scales to 8k+
  patients): **Pearson** correlation for continuous modalities (CNV,
  expression), **Jaccard** similarity for the binary mutation view; top-k edges
  retained per node.

These similarities are recomputed per fold from training-normalized features.

---

## 8. Labels & prediction target

- **Features (X):** the omics matrices only (CNV, mutation, and — in 3-modal —
  expression). Clinical columns (`age`, `gender`, `cancer_type`) are present in
  the survival file but are **not** used as inputs; `cancer_type` is retained
  only for post-hoc per-cancer C-index breakdowns.
- **Target (y):** the pair `(OS.time, OS)` — duration and event indicator —
  consumed by the Cox partial-likelihood loss (never fed to the network).
- **Metric:** Harrell's C-index (higher predicted risk ⇒ shorter survival).

---

## 9. Summary of filtering steps

| Filter | Criterion | Effect |
|--------|-----------|--------|
| Gene intersection | gene present in all active modalities | common gene axis (23384 → 17940 with expression) |
| Follow-up time | `OS.time > 0` | removes uninformative durations |
| Patient completeness | has all modalities + valid survival | 8755 (2-modal) / 8454 (3-modal) patients |
| Missing values | per-gene mean imputation | removes 1.34M expression NaNs |
| Pathway gene set | gene ∈ curated pathway (P-NET/Pathformer) | sparse biologically-masked inputs |
| Pathway size | ≥5 genes per pathway (PathCNN) | drops tiny pathways before PCA |

All steps are deterministic (seed 42) and shared across every baseline via
`pan_survival_data.py` and `survival_metrics.py`.
