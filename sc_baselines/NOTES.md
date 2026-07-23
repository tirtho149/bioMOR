# Single-cell baselines for bioMOR — integration notes

Three upstream single-cell cell-type-annotation models wired into the bioMOR
benchmark in "DeePathNet style": a self-contained upstream clone + a thin
`scripts/<name>_cv.py` CV runner that consumes the shared `biomor_common` backbone
(seed-42 CV5 folds, macro-F1 metric, common `scores_<stamp>.csv` schema) so every
baseline is apples-to-apples with bioMoR.

Datasets (all 7 gene-annotated AnnData under
`RecusrsiveQFormer/data/singlecell_resourced/<ds>/adata.h5ad`):
Baron, Lung, Oesophagus, Segerstolpe, Spleen, Tcell, Xin.

Each runner: `bc.load_sc(ds)` -> X/y/genes, `bc.load_sc_folds(ds,y)` -> 5 folds,
train on train+val, predict test per fold, `bc.write_scores(...)`. Scores land in
`sc_baselines/<name>/work_dirs/<ds>/scores_*.csv`.

## CIForm (`CIForm/scripts/ciform_cv.py`)
- Env: shared `/work/mech-ai-scratch/tirtho/.venv` (torch 2.10). Pure torch.
- Reuses upstream gene->sub-vector embedding (getXY: split gene vector into
  length-`gap` sub-vectors, per-subvector z-score), PositionalEncoding, and the
  `CIForm` transformer verbatim. log1p + train-fold HVG(2000) preprocessing.
  gap=1024 (nhead auto-adjusted to divide gap).
- Runner-level detail: CIForm's own module file runs a demo at import, so the two
  tiny reusable classes (PositionalEncoding, CIForm) are inlined byte-identically in
  the runner to avoid the demo side effects.

## TOSICA (`TOSICA/scripts/tosica_cv.py`)
- Env: shared `/work/mech-ai-scratch/tirtho/.venv` (torch 2.10 + einops).
- Reuses upstream `scTrans_model` (pathway-masked ViT) verbatim. The upstream
  package `__init__.py` pulls in `train.py -> torch.utils.tensorboard -> tensorboard`
  (not installed), so the runner loads `TOSICA_model.py` + `customized_linear.py`
  directly by file path, bypassing the package init. Upstream model code untouched.
- No `.gmt` pathway resources are bundled in this checkout, so we use TOSICA's own
  `gmt_path=None` fallback: a random-binomial gene->token mask ("Full connection!"
  path, author-supported). Architecture identical; only the pathway
  interpretability of tokens is dropped (irrelevant to the F1 benchmark).

## scTransSort (`scTransSort/scripts/sctranssort_cv.py`)
- Env: isolated `/work/mech-ai-scratch/tirtho/.venvs_baselines/scTransSort`
  (TensorFlow 2.15.1 + CUDA; scanpy/anndata/sklearn/pandas), built via
  `scripts/setup_venv.sbatch`. Upstream is TensorFlow/Keras so a separate venv is
  required; the shared venv has no TF.
- Reuses upstream `VisionTransformer` (Keras ViT) from the extension-less file
  `model/trans_model`, loaded via importlib. Reproduces the upstream gene->image
  embedding (`model/read` changefeature): each cell's gene vector is zero-padded and
  reshaped to an L x L x 3 image (L=ceil(sqrt(G))), classified by the ViT.
- Upstream's `vit_base_*` helpers hardcode img_size=224 (mismatched to our gene
  counts), but the `VisionTransformer` class is parametric on (img_size, patch_size),
  so the runner instantiates it directly with img_size=L and a patch_size dividing L.
  Only config is chosen; model code is untouched.

## SLURM
Scavenger partition, self-activating venv inside each script. Per baseline:
`scripts/smoke.sbatch` (Xin, 1 fold, tiny epochs) then `scripts/full_array.sbatch`
(array 0-6 over the 7 datasets, full epochs). Logs in `../logs/`.
