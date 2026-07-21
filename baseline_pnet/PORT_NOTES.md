# baseline_pnet — Port Notes

A log of every change made when slimming and modernizing the upstream
`pnet_prostate_paper` repo for use as a benchmark alongside
`main_soft_masking_2modal.py`.

- **Source repo:** https://github.com/marakeby/pnet_prostate_paper (`master`)
- **Source state:** Python 2.7 / TensorFlow 1.12 / Keras 2.2.4, 261 files, ~13 MB
- **Target state:** Python 3.10+ / TensorFlow 2.12+ / tf.keras, 43 files, 706 KB
- **Scope:** keep architecture + training scaffolding + the evaluation
  metrics you need (acc, precision, recall, auc, f1, f1_macro, aupr) +
  per-patient `y_score` for AUROC/AUPR plotting. Everything else dropped.

---

## 1. Cleanup — files removed

### Whole directories
| Directory | Reason |
|---|---|
| `analysis/` (1.9 MB, 53 .py) | Paper figure-generation scripts; not needed for benchmarking |
| `_plots/` (3.4 MB) | Pre-rendered figures from the published paper |
| `review/` (3.7 MB, 26 .py) | Paper reviewer-response extras (extra ablations, etc.) |
| `deepexplain/` | TF1-graph-mode interpretation library; not portable to TF2 |
| `data/prostate_paper/` | Prostate-specific data loaders + download scripts. We're using BRCA/pancancer, not the prostate data |
| `train/params/P1000/{compare,dense,external_validation,ML_params_search,number_samples,review}` | Non-P-NET baseline configs from the original paper. Kept only `train/params/P1000/pnet/` |

### Individual files
| File | Reason |
|---|---|
| `utils/plots.py` | You said you'll make AUROC/AUPR figures yourself from `y_score` |
| `utils/stats_utils.py`, `utils/stats_utils_delong_xu.py` | DeLong test + bootstrap CIs — not core metrics |
| `*_test.py` (all subdirs) | Module-level smoke tests, not needed |
| `.gitignore` | Not needed in stripped repo |
| `model/builders/builder_utils.py` | **Dead code** — no importer in the codebase. Used Theano + deprecated `keras.layers.merge`. Killing it eliminated the entire Theano + `merge` problem in one shot. (Note: the still-used file is `builders_utils.py` with the extra `s`.) |

---

## 2. Python 2 → Python 3 (mechanical)

Ran `2to3 -w -n .` over all remaining `.py` files. The auto-converter fixed:
- `print x` → `print(x)` (e.g. `utils/evaluate.py:42`, `model/nn.py:169,185`,
  `model/constraints_custom.py:23`, plus ~20 more)
- Dictionary `.iteritems()` / `.itervalues()` removals where present
- Other minor Py2 idioms

No `print [^(]` survivors remain after the pass.

---

## 3. Keras 2.2 → tf.keras (modern Keras 3 via TF 2.12+)

### Bulk import substitution

```sed
s|^from keras\.|from tensorflow.keras.|
s|^from keras import|from tensorflow.keras import|
s|^import keras\.|import tensorflow.keras.|
s|^import keras$|from tensorflow import keras|
```

Applied to every `.py` file. Affected lines across the repo:
- `model/{constraints_custom, callbacks_custom, nn, layers_custom, coef_weights_utils, model_utils}.py`
- `model/builders/{prostate_models, builders_utils}.py`
- `utils/evaluate.py`

### Path corrections (Keras 1 → Keras 3 layouts)

Done by additional sed pass:

| Old (Keras 2.2 / 1.x) | New (Keras 3 / tf.keras) |
|---|---|
| `from tensorflow.keras.engine import Layer` | `from tensorflow.keras.layers import Layer` |
| `from tensorflow.keras.engine import Model` | `from tensorflow.keras.models import Model` |
| `from tensorflow.keras.engine import InputLayer` | `from tensorflow.keras.layers import InputLayer` |
| `from tensorflow.keras.layers.core import X` | `from tensorflow.keras.layers import X` |

### Removed deprecated API
- `from tensorflow.keras.layers import Dense, merge` — `merge()` was removed in TF2. The only file that imported it (`model/builders/builder_utils.py`) was already dead code (no importer); deleted in Cleanup step.

### Keras 1 attribute `_keras_shape` → `.shape`
In `model/layers_custom.py`, the custom `Diagonal` and `SparseTF` layer
classes accessed input tensor shape via the Keras 1 private attribute
`x._keras_shape`. Replaced with `x.shape` (TF2-compatible). Affected lines:

- `model/layers_custom.py:155` — `n_features = x._keras_shape[1]` → `x.shape[1]`
- `model/layers_custom.py:156, 402` — `print('input dimensions {}'.format(x._keras_shape))` → `... x.shape`
- `model/layers_custom.py:400` — same as 155

(Commented-out references at lines 524–526 left as-is for historical context.)

---

## 4. TF1 → TF2 specifics

### `tf.random.set_random_seed` → `tf.random.set_seed`
- `train/run_me.py:25` — single replacement.

### `import imp` → `importlib`
The original `train/run_me.py` used Python 2's `imp.load_source(name, path)`
to dynamically load parameter files. `imp` was removed in Python 3.12.

**Replacement:** added a helper near the top of `train/run_me.py`:

```python
import importlib.util as _il

def _load_module_from_path(name, path):
    """Modern replacement for the deprecated `imp.load_source`."""
    spec = _il.spec_from_file_location(name, path)
    mod = _il.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
```

And rewrote the call site:
- Old: `params = imp.load_source(params_file, params_file_full)`
- New: `params = _load_module_from_path(params_file, params_file_full)`

### `tf.Session()` removal
Only one usage — `model/coef_weights_utils.py::get_deep_explain_score_layer`.
That function was tied to the (now-removed) DeepExplain TF1 library.
**Stubbed** the function body with a clear `NotImplementedError` and a
docstring explaining the TF2 port limitation. Functions are guarded — the
slim port sets `feature_importance: None` in the active param files, so
this stub is never reached at runtime.

```python
def get_deep_explain_score_layer(...):
    """DeepExplain-based interpretation. NOT ported to TF2.
    ...
    Set `feature_importance: None` in your param file to skip this entirely.
    """
    raise NotImplementedError(
        "get_deep_explain_score_layer is disabled in the TF2 port. ..."
    )
```

---

## 5. Functional patches (your specific asks)

### Added `f1_macro` to evaluation metrics
**File:** `utils/evaluate.py`

Added `metrics.f1_score(y_test, y_pred, average='macro')` to both
`evalualte()` and `evalualte_classification_binary()`. The returned `score`
dict now contains: `accuracy, precision, recall, auc, f1, f1_macro, aupr`.

The logging line was updated to print `f1_macro` alongside the others.

### `y_score` per-patient export — already there ✓
`pipeline/crossvalidation_pipeline.py:99` writes `info['pred_score'] =
y_pred_score` and then `info.to_csv(file_name)` for each fold. Same in
`pipeline/LeaveOneOut_pipeline.py:85`. No code change needed.

Per-fold CSVs land in the run's output directory and look like:
```
sample_id  pred  pred_score  y_true
P001       1     0.834       1
P002       0     0.121       0
...
```
This is exactly what you'd feed into a separate plotting script for
AUROC/AUPR figures.

### `data/data_access.py` — removed prostate dependency
The original imported `from data.prostate_paper.data_reader import
ProstateDataPaper` at module top, and only supported `data_type ==
'prostate_paper'`. Since we deleted `data/prostate_paper/`, that import
would fail.

**Replacement:** removed the bad import, replaced the dispatch with a
clear `NotImplementedError` + a template comment block showing how to
register a new data reader (`BRCADataReader`, `PancancerDataReader`, etc).
You add the `if self.data_type == 'brca':` branch yourself when you
write the reader.

### Param files — `feature_importance: None`
Set in all three `train/params/P1000/pnet/*.py` so the
NotImplementedError-stub above is never reached:
- `onsplit_average_reg_10_tanh_large_testing.py:68`
- `onsplit_average_reg_10_tanh_large_testing_inner.py:69`
- `crossvalidation_average_reg_10_tanh.py:68`

If you ever port the deepexplain machinery to TF2 (using `tf.GradientTape`),
revert these settings.

### `environment.yml` — fully unpinned
Old: `python=2.7.15`, `tensorflow=1.12.0`, `keras=2.2.4`, pandas=0.23.4, …
(every package version-locked to a 2019 stack).

New: `python>=3.10`, `tensorflow>=2.12`, no other version pins. Use:

```bash
conda env create -f environment.yml -n pnet
conda activate pnet
```

or with pip:
```bash
pip install 'tensorflow>=2.12' numpy pandas scipy scikit-learn \
            h5py pyyaml matplotlib seaborn networkx tqdm openpyxl xlrd \
            lifelines
```

---

## 6. Validation

After all changes, ran `python -m py_compile` on every remaining .py file:

```
=== compile errors: 0 (out of 38 files) ===
```

Final sweep for legacy patterns:

| Pattern | Result |
|---|---|
| Python 2 `print x` (no parens) | none |
| `import imp` | none |
| `tf.placeholder`, `tf.Session()`, `tf.global_variables`, `tf.contrib`, `K.set_session` | none in live code (one match is inside the docstring of the stubbed function) |
| `_keras_shape` (live code) | none |
| `K.tf` direct access | none |

---

## 7. File manifest (post-port)

```
baseline_pnet/                            706 K total, 43 files
├── HOW_TO_RUN.txt                        — runbook
├── PORT_NOTES.md                         — this file
├── README.md                             — upstream README (unmodified)
├── LICENSE                               — upstream license
├── environment.yml                       — fully unpinned for py3.10/TF2
├── config_path.py                        — paths config
│
├── data/
│   ├── __init__.py
│   ├── data_access.py                    — PATCHED: prostate import removed,
│   │                                       stub for new data readers
│   ├── gmt_reader.py
│   └── pathways/
│       ├── __init__.py
│       ├── gmt_pathway.py
│       ├── reactome.py
│       └── pathways_short_names.xlsx
│
├── model/
│   ├── __init__.py
│   ├── callbacks_custom.py               — imports rewritten to tf.keras
│   ├── coef_weights_utils.py             — get_deep_explain_score_layer stubbed
│   ├── constraints_custom.py             — imports rewritten
│   ├── layers_custom.py                  — _keras_shape → .shape (Diagonal, SparseTF)
│   ├── model_factory.py
│   ├── model_utils.py                    — imports rewritten
│   ├── nn.py                             — imports rewritten
│   ├── pathway_connection.py
│   └── builders/
│       ├── __init__.py
│       ├── builders_utils.py             — imports rewritten
│       └── prostate_models.py            — imports rewritten
│       (NOTE: builder_utils.py — singular — was deleted; was dead code)
│
├── pipeline/
│   ├── __init__.py
│   ├── crossvalidation_pipeline.py       — already writes y_score per fold ✓
│   ├── LeaveOneOut_pipeline.py           — already writes y_score per fold ✓
│   ├── one_split.py
│   ├── pipe_utils.py
│   └── train_validate.py
│
├── preprocessing/
│   ├── __init__.py
│   └── pre.py
│
├── train/
│   ├── __init__.py
│   ├── run_me.py                         — tf.random.set_seed, importlib helper
│   └── params/P1000/pnet/
│       ├── crossvalidation_average_reg_10_tanh.py     — feature_importance: None
│       ├── onsplit_average_reg_10_tanh_large_testing.py   — feature_importance: None
│       └── onsplit_average_reg_10_tanh_large_testing_inner.py — feature_importance: None
│
└── utils/
    ├── __init__.py
    ├── evaluate.py                       — PATCHED: f1_macro added
    ├── loading_utils.py
    ├── logs.py
    ├── rnd.py
    └── saving.py
```

---

## 8. What's NOT done (and why)

| Task | Status | Why |
|---|---|---|
| Write a BRCA/pancancer data reader | **DONE** (see §10) | `data/brca/data_reader.py` + registered in `data/data_access.py` |
| Pathway file converter | **DONE** (see §10) | `preprocessing/build_reactome_files.py` |
| Apply SKILLS.md hyperparam alignment | **DONE** (see §10) | active param file updated |
| Download Reactome `.gmt` files | **OPTIONAL** | Public Reactome download from https://reactome.org/download-data into `_database/pathways/Reactome/`. Only needed if you want P-NET's full deep pathway hierarchy. The converter produces a flat 1-level tree from your `filtered_pathways.csv` which is enough to train, but loses P-NET's depth advantage |
| Port DeepExplain to TF2 / GradientTape | **NOT DONE** | Out of scope; only metrics + y_score requested. Stubbed cleanly |
| End-to-end runtime test of `train/run_me.py` | **NOT DONE** | This env has no TF installed; only static `py_compile` + module-import checks were run |

---

## 9. Most likely runtime issues (when you actually run it)

Honest list of things I'd expect to break first when this hits an actual
TF2 environment:

1. **Layer-shape semantics in `Diagonal` and `SparseTF`.** Keras 1's
   `_keras_shape` was eagerly populated by the framework; TF2's `.shape`
   may be a `TensorShape` with `None` entries during graph tracing.
   The `n_features = x.shape[1]` line might need `n_features =
   K.int_shape(x)[1]` instead. Easy fix once you hit it.

2. **`K.bias_add` / `K.reshape` style.** These backend functions still
   exist in `tf.keras.backend` but some signatures shifted slightly
   between Keras 2.2 and Keras 3. If `bias_add` complains, swap to
   `tf.nn.bias_add` or `+ self.bias` directly.

3. **Custom callbacks in `model/callbacks_custom.py`.** Some Keras 2
   callback hooks (`on_epoch_end`, etc.) have slightly different
   signatures in Keras 3. May need minor tweaks.

4. **Sample weight / class weight handling.** The param files set
   `class_weight = {0: 0.75, 1: 1.5}` — that should still work in
   TF2/Keras 3 unchanged, but worth verifying.

5. **`pipeline/train_validate.py`** uses `valid_mut_df = genes_df.merge(...)`
   — that's `pandas.DataFrame.merge`, NOT the deprecated `keras.layers.merge`,
   so it's fine. (Flagging because it might look suspicious at first
   glance.)

If any of these hit you, the trace + filename will tell us where to patch.

---

## 10. Adaptation phase (post-port; data + config wiring)

After the porting phase (§1–§7) made the codebase parseable on Python 3 /
TF 2, three more things were needed to actually run on BRCA/pancancer data:

### 10a. BRCA/pancancer data reader

**Added file:** `data/brca/data_reader.py`

Defines `BRCADataReader`, a drop-in replacement for the deleted prostate
reader. Takes the standard Graph_Transformer CSVs:

    mutation_data.csv  — patients × genes, binary
    cnv_data.csv       — patients × genes, continuous
    <labels_filename>  — two cols (patient_id, label)

Returns:

- `x`: ndarray of shape `(N_patients, 2 × N_genes)`, **gene-grouped** layout
  `[g0_mut, g0_cnv, g1_mut, g1_cnv, …]`. This ordering is REQUIRED by
  `Diagonal`'s per-node reshape (`n_inputs_per_node = n_features / n_genes`).
- `y`: int labels in `{0, 1}`. Normalizes both numeric and
  `Primary`/`Metastatic` string labels.
- `info`: list of patient IDs.
- `columns`: `pandas.MultiIndex` with levels `[genes, ['mut','cnv']]` so
  `prostate_models.py::build_pnet2` can do `cols.levels[0]` to extract the
  gene list it passes to `get_pnet`.
- Single stratified train/val/test split: 80/10/10 by default,
  `random_state=42`. CNV is z-scored using **train statistics only**.

**Patched file:** `data/data_access.py`

The dispatcher in `Data.__init__` now branches on `data_type == 'brca'` and
imports `BRCADataReader`. Add more branches the same way for new datasets.
Verified by import smoke-test:

```
Data(id='BRCA', type='brca', params={'data_dir': '/nonexistent'})
→ FileNotFoundError: /nonexistent/mutation_data.csv
(dispatch reached the reader; failed expectedly on missing file)
```

### 10b. Pathway converter (filtered_pathways.csv → Reactome layout)

**Added file:** `preprocessing/build_reactome_files.py`

CLI script that reads `filtered_pathways.csv` (columns `Pathway_ID,
Pathway_Name, Genes`) and emits the three files P-NET's `ReactomeNetwork`
expects under `_database/pathways/Reactome/`:

| Output file | Content |
|---|---|
| `ReactomePathways.txt` | `reactome_id <TAB> pathway_name <TAB> species` |
| `ReactomePathways.gmt` | `pathway_name <TAB> pathway_id <TAB> descrip <TAB> gene1 <TAB> gene2 …` (loader reads `pathway_col=1, genes_col=3`) |
| `ReactomePathwaysRelation.txt` | `child <TAB> parent` |

Pathway IDs are prefixed with `HSA-` if they don't already contain it,
so they pass `ReactomeNetwork`'s species filter
(`hierarchy[hierarchy['child'].str.contains('HSA')]`).

**Topology produced** (because your `filtered_pathways.csv` is FLAT):

```
       root  (added by P-NET)
        │
   ┌────┼────┬────┬────┐
   ▼    ▼    ▼    ▼    ▼
  P_1  P_2  P_3 ... P_N   (your pathways; in_degree=0 in code's child→parent
                          direction, so they're direct children of root)
   │    │    │    │    │
   ▼    ▼    ▼    ▼    ▼
 (HSA-DUMMY-TARGET, sits outside ego_graph(root, radius=1))
```

With `n_hidden_layers=1` (set in the param file), the architecture is:

```
inputs (2 × n_genes)
   ↓  Diagonal h0 (gene-wise aggregation)
genes (n_genes units)
   ↓  SparseTF h1 (gene→pathway mask from your filtered_pathways.csv)
pathways (n_pathways units)
   ↓  Dense decision outputs (one per layer for deep supervision)
```

Your pathway data has NO parent-child relations, so this is a single-layer
sparse model. P-NET's distinctive depth advantage is lost. To recover it,
download Reactome's native files (`ReactomePathways.gmt`,
`ReactomePathwaysRelation.txt`, `ReactomePathways.txt`) from
https://reactome.org/download-data and place them in the same output dir,
then bump `n_hidden_layers` accordingly.

**Usage:**
```bash
python preprocessing/build_reactome_files.py \
    --pathways /…/Graph_Transformer/data/brca/filtered_pathways.csv \
    --out      _database/pathways/Reactome
```

### 10c. SKILLS.md hyperparameter alignment

**Patched file:** `train/params/P1000/pnet/onsplit_average_reg_10_tanh_large_testing.py`

Changed:

| Field | Before | After | Reason |
|---|---|---|---|
| `data_base['type']` | `'prostate_paper'` | `'brca'` | Use new reader |
| `data_base['params']` | prostate-specific keys | `data_dir, labels_filename, val_size, test_size, random_state, zscore_cnv` | New reader interface |
| `n_hidden_layers` | 5 | 1 | Flat pathway data |
| `base_dropout` | 0.5 | 0.2 | SKILLS.md |
| `loss_weights` | `[2,7,20,54,148,400]` | `[2,7,20]` | Now 3 outputs (was 6) |
| `epoch` | 300 | 200 | SKILLS.md |
| `batch_size` | 50 | 16 | SKILLS.md |
| `lr` | 1e-3 | 1e-4 | SKILLS.md |
| `select_best_model` | False | True | Standard early-stop pattern |
| `monitor` | `val_o6_f1` | `val_o2_f1` | matches new top-layer output |
| `early_stop` | False | True | SKILLS.md |
| `class_weight` | `'auto'` | `'auto'` | Unchanged (balanced) |
| `optimizer` | `'Adam'` | `'Adam'` (with switch comment) | String left; user can swap to `'AdamW'` in TF 2.11+ or pass instance |
| secondary logistic model | included | removed | Hardcoded prostate class weights; replaced with a comment showing how to re-add |

**Why optimizer stayed `'Adam'`:** Switching to `'AdamW'` would require
either a TF-side import in the param file (which would fail on
`py_compile` in any env without TF) or a string that depends on
Keras 3.x. Left a comment with two ways to switch — user picks.

**Why I added `utils/plots.py` back as a no-op stub:** Multiple pipeline
modules import `from utils.plots import …` at the top level. Removing the
file (per §1) broke imports. The replacement is a stub that wires every
plotting function to a no-op so imports succeed but no figures are
generated. You already said you'll make AUROC/AUPR figures yourself from
`pred_score` columns.

### Files added/modified in this phase

```
data/brca/__init__.py                 (new, empty)
data/brca/data_reader.py              (new, ~200 lines — BRCADataReader)
data/data_access.py                   (patched — added 'brca' dispatch)
preprocessing/build_reactome_files.py (new, ~200 lines — pathway converter)
utils/plots.py                        (restored as no-op stub)
train/params/P1000/pnet/onsplit_average_reg_10_tanh_large_testing.py
                                      (patched per SKILLS.md + data-reader
                                       interface + n_hidden_layers=1)
```

### Verification

```
$ python -m py_compile <every .py file>
errors: 0 / 42 files

$ python data/brca/data_reader.py     # module import
BRCADataReader OK

$ python preprocessing/build_reactome_files.py --help
(usage text displayed correctly)

$ Data(type='brca', params={'data_dir': '/nonexistent'})
FileNotFoundError: /nonexistent/mutation_data.csv     # dispatch works
```

No runtime training was attempted (no TF installed here). The list of
likely-runtime issues in §9 still applies.

---

## Quick references

- Cleanup + porting log: this file
- Runbook (how to invoke): `HOW_TO_RUN.txt`
- Hyperparameter alignment recipe: `../SKILLS.md` (covers all baselines)
- Original repo (for restoring deleted bits if needed):
  `/lustre/hdd/LAS/weile-lab/howlader/GraphPath_baselines/pnet_prostate_paper-master.zip`
  — wait, that's deleted. Re-download from
  https://github.com/marakeby/pnet_prostate_paper if you need the originals.
