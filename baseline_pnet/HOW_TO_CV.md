# P-NET 10-fold CV recipe (reusable for any dataset)

This is the standard recipe for running P-NET as a 10-fold cross-validation
baseline on a 2-modality (mutation + CNV) TCGA-style dataset. It works for
BRCA, BLCA, pan_meta_pri, pan_brca_molsubtype, kirc_molsubtype, etc. — any
folder that contains `mutation_data.csv`, `cnv_data.csv`, and a labels CSV
in the layout the `BRCADataReader` expects.

Outputs you get for every run (no extra scripts needed):
- Per-fold per-patient predictions WITH `pred_score` (the y_score for AUROC /
  AUPR plotting): `<model>_testing_fold_<i>.csv` + `<model>_traing_fold_<i>.csv`
- Per-fold metric table:                                `folds.csv`
- Per-fold + mean ± std summary CSV:                    `<model>_scores.csv`
- Per-fold + mean ± std summary (JSON inside YAML):     `<model>_params.yml`
- Per-fold saved Keras weights:                         `fs/<model>_<i>.h5`

Metrics computed each fold: `accuracy, precision, recall, auc, f1, f1_macro,
aupr`.

------------------------------------------------------------------------
Step 1 — Make sure your data directory is in the right layout
------------------------------------------------------------------------
The dataset directory must contain:

    <data_dir>/
        mutation_data.csv     patient_id, gene1, gene2, …  (binary 0/1)
        cnv_data.csv          patient_id, gene1, gene2, …  (continuous, z-scored
                                                           by train-stats inside)
        <labels_filename>     patient_id, label[, …]       (label = 0/1 or
                                                           "Primary"/"Metastatic";
                                                           only the FIRST label
                                                           column is used)

Patients/genes are intersected across the three files; you don't need them
pre-aligned. The `selected_genes_filename` whitelist (P-NET's default cancer-
gene list) further restricts the feature set.

------------------------------------------------------------------------
Step 2 — Edit the CV param file to point at your dataset
------------------------------------------------------------------------
File: `train/params/P1000/pnet/crossvalidation_average_reg_10_tanh.py`

Change ONLY the `data_base` block:

    data_base = {'id': '<SHORT_ID>',           # e.g. 'BRCA', 'PANCANCER_PRI_MET'
                 'type': 'brca',                # always 'brca' (= BRCADataReader)
                 'params': {
                     'data_dir': '<absolute path to dataset folder>',
                     'labels_filename': 'patient_labels.csv',   # or whatever yours is named
                     'selected_genes_filename':
                         '/lustre/hdd/LAS/weile-lab/howlader/GraphPath/p_net_data/tcga_prostate_expressed_genes_and_cancer_genes.csv',
                     'val_size': 10 / 90,       # inner-fold val split (matches main_soft_masking)
                     'test_size': 0.1,          # unused by CV (the K-fold split overrides it)
                     'random_state': 42,
                     'zscore_cnv': True,
                 }
                 }

Don't touch the rest of the file — the architecture and SKILLS.md-aligned
hyperparameters are already set:
    n_hidden_layers   = 5             (full Reactome hierarchy)
    base_dropout      = 0.2
    epoch             = 200
    batch_size        = 16
    lr                = 1e-4
    optimizer         = 'Adam'        (switch to 'AdamW' in the file for strict parity)
    early_stop        = True
    select_best_model = True          (monitor = val_o6_f1)
    class_weight      = 'auto'        (balanced)
    n_splits          = 10            (stratified K-fold, shuffle, random_state=123)

To use a different gene whitelist or none at all, change/remove
`selected_genes_filename`.

------------------------------------------------------------------------
Step 3 — Make sure run_me.py picks the CV param file
------------------------------------------------------------------------
File: `train/run_me.py`

The `params_file_list` block should have ONLY this line uncommented:

    params_file_list.append('./pnet/crossvalidation_average_reg_10_tanh')

If you want the single-split file instead, uncomment that line and comment
this one out.

------------------------------------------------------------------------
Step 4 — Run training
------------------------------------------------------------------------
    cd /lustre/hdd/LAS/weile-lab/howlader/GraphPath_baselines/baseline_pnet
    conda activate pnet
    python train/run_me.py

Or with full logging to a file:

    mkdir -p _logs/<dataset>_runs
    nohup python train/run_me.py \
        > _logs/<dataset>_runs/<dataset>_cv10_$(date +%Y%m%d_%H%M%S).log 2>&1 &

------------------------------------------------------------------------
Step 5 — Read the outputs
------------------------------------------------------------------------
Output directory:

    _logs/p1000/./pnet/crossvalidation_average_reg_10_tanh/
        ├── P-net_<SHORT_ID>_scores.csv        ← per-fold + mean + std (the table you want)
        ├── P-net_<SHORT_ID>_params.yml        ← params + scores JSON
        ├── P-net_<SHORT_ID>_testing_fold_0.csv  ← per-patient y_score, fold 0
        ├── P-net_<SHORT_ID>_traing_fold_0.csv   ← train y_score, fold 0
        ├── …                                    (one pair per fold, 0..9)
        ├── folds.csv                          ← raw per-fold metrics across models
        ├── fs/                                ← per-fold .h5 weights
        └── log.log                            ← training log

`<SHORT_ID>` is what you set in `data_base['id']` (Step 2). Existing runs
write into the SAME directory, so if you re-run the same dataset twice the
prediction CSVs get overwritten but the `.keras` checkpoints stack up.

Per-patient prediction CSV columns:
    , pred, pred_score, y
    TCGA-XX-…, 0, 0.123, 0
    TCGA-YY-…, 1, 0.871, 1

`pred_score` is what you feed into AUROC / AUPR plotting (the y_score).
`pred` is the thresholded class.

`<model>_scores.csv` columns: `accuracy, precision, recall, auc, f1, f1_macro,
aupr`. Rows: `fold_0 … fold_9`, then `mean`, then `std`.

------------------------------------------------------------------------
Step 6 — (only first time per dataset) Build the Reactome files
------------------------------------------------------------------------
This is a one-time prerequisite. If `_database/pathways/Reactome/` already
has `ReactomePathways.gmt`, `ReactomePathways.txt`, and
`ReactomePathwaysRelation.txt`, skip this step.

To regenerate using the real Reactome hierarchy (recommended; matches the
current state of `_database/pathways/Reactome/`):

    python preprocessing/build_reactome_files.py \
        --pathways  /lustre/hdd/LAS/weile-lab/howlader/Graph_Transformer/data/reactome_latest/filtered_pathways_curated.csv \
        --relations /lustre/hdd/LAS/weile-lab/howlader/Graph_Transformer/data/reactome_latest/HumanPathwaysRelation.csv \
        --out       _database/pathways/Reactome

Or, with a flat pathway list (no relations):

    python preprocessing/build_reactome_files.py \
        --pathways /…/filtered_pathways.csv \
        --out      _database/pathways/Reactome

If you use the flat form, also set `n_hidden_layers=1` in the param file
to match the single-layer topology. Hierarchical mode keeps the default
`n_hidden_layers=5` (pnet pads shallower branches with `_copy` nodes).

------------------------------------------------------------------------
Quick switching cheat-sheet
------------------------------------------------------------------------
| Dataset            | data_dir (relative to /…/Graph_Transformer/data*)       | id                    |
|--------------------|---------------------------------------------------------|-----------------------|
| BRCA               | data/brca                                               | BRCA                  |
| BLCA               | data/blca                                               | BLCA                  |
| Pan primary/met    | data_tcga/pan_meta_pri                                  | PANCANCER_PRI_MET     |
| Pan BRCA molsub    | data_tcga/pan_brca_molsubtype                           | PAN_BRCA_MOLSUB       |
| KIRC molsubtype    | data_tcga/kirc_molsubtype                               | KIRC_MOLSUB           |
| Pan survival       | data_tcga/pan_survival                                  | PAN_SURVIVAL          |
| Pan 3-modal*       | data_tcga/pan_3modal                                    | PAN_3MODAL            |

*Note: P-NET as ported here is **2-modality only** (mut + cnv). The 3-modal
folder includes an expression CSV that the `BRCADataReader` will ignore.

To switch dataset: edit `data_base['id']` and `data_base['params']['data_dir']`
in `crossvalidation_average_reg_10_tanh.py`, then re-run `python train/run_me.py`.

That's it. The CV pipeline does the rest — splits, training, eval, y_score
export, mean±std summary.
