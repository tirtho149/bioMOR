# Datasets

Cancer-genomics datasets used to train and benchmark PATH (a pathway-based graph
transformer). Each dataset is one cancer cohort. Patients are described by one or
more genomic modalities (somatic mutation, copy-number variation, and — for 3-modal
datasets — gene expression) over a shared set of genes, and genes are grouped into
biological pathways that form the nodes of a pathway–pathway interaction graph.

All cohorts share the **same pathway graph**: 1,268 Reactome pathways and an
identical 1,268 × 1,268 adjacency matrix. They differ in the patient cohort, the
gene feature space, and the prediction task.

## Files in each dataset

Every dataset directory contains a genomic modality matrix per modality, a
pathway/graph definition, and a target file. Not all files appear in every dataset
(expression only in 3-modal datasets; survival datasets use `survival.csv` instead
of `labels.csv` and keep the older `filtered_pathways.csv` name).

| File | Shape | Contents |
|---|---|---|
| `mutation_data.csv` | patients × genes | Somatic mutation matrix. Rows = patients, columns = genes. Binary {0,1} = mutated / wild-type for most cohorts (prostate holds mutation **counts**, 0–12). Row index = patient ID, header row = gene symbols. |
| `cnv_data.csv` | patients × genes | Copy-number variation matrix. Same patients × genes layout as `mutation_data.csv`. GISTIC thresholded values {−2,−1,0,1,2} = deep loss / loss / neutral / gain / amplification. |
| `expression_data.csv` | patients × genes | Gene-expression matrix (**3-modal datasets only**). Same patients × genes layout; log-transformed RNA-seq expression values. Gene columns match `mutation_data.csv`/`cnv_data.csv`. |
| `labels.csv` | patients × 2 (or 4) | Classification labels. Column 1 = patient ID, column 2 = `response` (the target). `pan_meta_pri*` also have `sample_type` and `primary_disease`. |
| `survival.csv` | patients × 6 | Survival target (**survival datasets only**, replaces `labels.csv`). Columns: `id`, `OS` (event 0/1), `OS.time` (days), `cancer_type`, `age`, `gender`. |
| `pathways.csv` / `filtered_pathways.csv` | 1,268 × 3 | Pathway definitions. Columns: `Pathway_ID` (Reactome ID, e.g. `R-HSA-109581`), `Pathway_Name` (e.g. `Apoptosis`), `Genes` (comma-separated member gene symbols). These 1,268 pathways are the graph nodes. Survival datasets keep the older `filtered_pathways.csv` name. |
| `adjacency_matrix.csv` | 1,268 × 1,268 | Pathway–pathway interaction graph. Square matrix; entry (i,j) = interaction strength between pathway i and pathway j. Row/column order matches the pathways file. Same file across all cohorts. |

Within a cohort, all modality matrices cover the **same** patient set and the
**same** gene columns (align by patient ID).

## Datasets and tasks

| Dataset | Modalities | Patients | Genes | Task / label meaning | Classes / target distribution |
|---|---|---|---|---|---|
| `blca` | mut, cnv | 404 | 23,384 | Bladder urothelial carcinoma, binary clinical outcome (TCGA primary tumors; `response` undocumented, likely early-vs-late stage) | 2 — 1: 273 (68%) / 0: 131 (32%) |
| `brca_5_class` | mut, cnv | 526 omics / 518 labeled | 40,543 | Breast carcinoma, 5-class label (`response` 0–4, likely molecular subtype) | 5 — 2:262, 3:112, 0:95, 1:35, 4:14 |
| `stad` | mut, cnv | 414 | 23,384 | Stomach adenocarcinoma, binary clinical outcome (TCGA primary tumors; likely early-vs-late stage) | 2 — 1: 228 (55%) / 0: 186 (45%) |
| `prostate` | mut, cnv | 1,011 | 8,434 | Prostate cancer, binary label. Non-TCGA cohort (IDs `AAPC-STID…-Tumor-SM-…`); mutation matrix holds counts, not binary | 2 — 0: 678 (67%) / 1: 333 (33%) |
| `pan_meta_pri` | mut, cnv | 8,893 | 23,384 | Pan-cancer **metastatic vs primary** classification across 32 cancer types. `response` 1 = Metastatic (361), 0 = Primary Tumor (8,532); extra `sample_type`, `primary_disease` | 2 — 0: 8,532 (96%) / 1: 361 (4%) — highly imbalanced |
| `pan_meta_pri_3modal` | **mut, cnv, expr** | 8,586 | 17,940 | 3-modal version of `pan_meta_pri` (met vs primary). Restricted to patients with all three modalities and to genes shared across modalities | 2 — 0: 8,225 (96%) / 1: 361 (4%) |
| `pan_survival_cox` | mut, cnv | 8,823 | 23,384 | Pan-cancer **overall-survival** prediction (Cox regression). Target in `survival.csv`: `OS` (event), `OS.time` (days), plus `cancer_type`, `age`, `gender` | survival — right-censored OS across cancer types |

### Notes

- **Only `pan_meta_pri*` have documented label semantics** (`sample_type` column
  confirms 1 = Metastatic, 0 = Primary Tumor). For `blca`, `stad`, and
  `brca_5_class` the `response` meaning is not stored in the files; sample IDs are
  TCGA primary-tumor barcodes (`-01`), so these are not metastasis tasks.
- **`pan_meta_pri_3modal`** is the aligned tri-modal cohort: all three matrices
  (mutation, CNV, expression) share the **same 17,940 genes in the same order** and
  the **same 8,586 patients**. It is a strict subset of `pan_meta_pri` — the 307
  dropped patients (relative to the 8,893) all lack expression and are **all
  Primary**, so no metastatic samples are lost.
- **`pan_survival_cox`** is a survival task, not classification: use `survival.csv`
  (`OS`/`OS.time`) with a Cox/partial-likelihood loss, and its pathway file is named
  `filtered_pathways.csv`.
- **`brca_5_class`** has 526 patients in the omics files but only 518 in
  `labels.csv` — 8 patients are unlabeled. Join on patient ID (inner join) when
  loading. It also has the largest gene space (40,543).
- **`prostate`** is the most distinct cohort: an external (non-TCGA) dataset, its
  mutation matrix stores mutation counts rather than binary flags, and mutation/CNV
  row order differs (align by ID).
- **`pan_meta_pri` / `pan_meta_pri_3modal`** are heavily class-imbalanced
  (~4% metastatic); use class weighting or focal loss.
- `blca` labels are stored as floats (`1.0`/`0.0`); cast to int before training.
