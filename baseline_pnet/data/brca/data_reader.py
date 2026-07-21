"""
BRCA / pancancer-style data reader for P-NET.

Mirrors the original ProstateDataPaper interface so the rest of P-NET can use
it without changes. Expects three CSVs in `data_dir`:

    mutation_data.csv   rows = patients, cols = genes, values = binary 0/1
    cnv_data.csv        rows = patients, cols = genes, values = continuous
    <labels_filename>   two cols: patient_id, label (binary 0/1 or
                        "Primary"/"Metastatic")

Patients and genes are intersected across files. Features are emitted in
gene-grouped (mut, cnv) pairs — REQUIRED layout for P-NET's Diagonal layer,
which reshapes input from (B, n_features) → (B, n_genes, n_inputs_per_node)
and sums along the per-node axis. With both modalities, n_inputs_per_node=2
and features must be ordered [g0_mut, g0_cnv, g1_mut, g1_cnv, …].

The `columns` returned to P-NET is a pandas MultiIndex with two levels
([genes, ['mut','cnv']]) so downstream code (`prostate_models.py` and
`get_layer_maps`) can extract the gene list via `cols.levels[0]`.

Stratified train/val/test split: 80/10/10 by default, seeded for
reproducibility — matches the scheme used by main_soft_masking_2modal.py.
"""
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _normalize_labels(label_series: pd.Series) -> pd.Series:
    """Coerce labels to int {0,1}. Accepts:
        - 0/1 ints or strings ("0", "1")
        - "Primary" / "Metastatic" (case-insensitive)
    """
    s = label_series.astype(str).str.lower().str.strip()
    mapping = {"primary": 0, "metastatic": 1, "p": 0, "m": 1,
               "0": 0, "1": 1, "false": 0, "true": 1}
    if s.isin(mapping.keys()).all():
        return s.map(mapping).astype(int)
    # Fall back: try plain int cast (handles already-numeric columns)
    try:
        return label_series.astype(int)
    except Exception as e:
        raise ValueError(
            "Could not coerce labels to int. Got values: {}. "
            "Expected 0/1 or Primary/Metastatic.".format(label_series.unique()[:10])
        ) from e


class BRCADataReader:
    """
    Drop-in replacement for the prostate-paper ProstateDataPaper reader,
    targeted at BRCA / pancancer-style two-modality (mutation + CNV) data.

    Parameters
    ----------
    data_dir : str
        Directory containing mutation_data.csv, cnv_data.csv, <labels_filename>.
    labels_filename : str
        Name of the labels CSV inside data_dir. Two columns: patient_id, label.
    val_size : float
        Fraction of train_val to hold out as validation. Default 10/90
        (so per-fold ratios are 80% train / 10% val / 10% test of total).
    test_size : float
        Fraction of total to hold out as test. Default 0.1.
    random_state : int
        Seed for the stratified splits. Default 42.
    zscore_cnv : bool
        If True, z-score CNV using train statistics only (recommended).
    mutation_filename, cnv_filename : str
        Override default filenames if your data uses different names.
    selected_genes_filename : str, optional
        Path to a CSV with one column of gene symbols; restricts the feature
        set to those genes. None = keep all genes that appear in both
        mutation and CNV CSVs.
    """

    def __init__(self,
                 data_dir: str,
                 labels_filename: str = "labels.csv",
                 val_size: float = 10/90,
                 test_size: float = 0.1,
                 random_state: int = 42,
                 zscore_cnv: bool = True,
                 mutation_filename: str = "mutation_data.csv",
                 cnv_filename: str = "cnv_data.csv",
                 selected_genes_filename: Optional[str] = None,
                 **_unused_kwargs):
        self.data_dir = data_dir
        self.labels_filename = labels_filename
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state
        self.zscore_cnv = zscore_cnv
        self.mutation_filename = mutation_filename
        self.cnv_filename = cnv_filename
        self.selected_genes_filename = selected_genes_filename

        self._load_tables()
        self._build_feature_matrix()
        self._stratified_split()

    # ------------------------------------------------------------------
    # data loading
    # ------------------------------------------------------------------
    def _load_tables(self):
        mut_path = os.path.join(self.data_dir, self.mutation_filename)
        cnv_path = os.path.join(self.data_dir, self.cnv_filename)
        lab_path = os.path.join(self.data_dir, self.labels_filename)
        for p in (mut_path, cnv_path, lab_path):
            if not os.path.exists(p):
                raise FileNotFoundError(p)

        mut_df = pd.read_csv(mut_path)
        cnv_df = pd.read_csv(cnv_path)
        lab_df = pd.read_csv(lab_path)

        # First column = patient ID, rest = gene features.
        mut_df = mut_df.set_index(mut_df.columns[0])
        cnv_df = cnv_df.set_index(cnv_df.columns[0])
        lab_df = lab_df.set_index(lab_df.columns[0])

        # Force string IDs everywhere so intersections behave deterministically.
        for df in (mut_df, cnv_df, lab_df):
            df.index = df.index.astype(str)

        # Optional gene whitelist
        if self.selected_genes_filename is not None:
            sel = pd.read_csv(self.selected_genes_filename, header=None)[0].astype(str).tolist()
            mut_df = mut_df[[g for g in sel if g in mut_df.columns]]
            cnv_df = cnv_df[[g for g in sel if g in cnv_df.columns]]
            logging.info("BRCADataReader: restricted to %d selected genes.", len(sel))

        # Intersect genes across modalities (preserve mutation order)
        common_genes = [g for g in mut_df.columns if g in cnv_df.columns]
        if not common_genes:
            raise ValueError("No genes overlap between mutation and CNV tables.")
        mut_df = mut_df[common_genes]
        cnv_df = cnv_df[common_genes]

        # Intersect patient IDs
        common_ids = (mut_df.index
                      .intersection(cnv_df.index)
                      .intersection(lab_df.index))
        if len(common_ids) == 0:
            raise ValueError("No patient IDs overlap between modalities and labels.")
        mut_df = mut_df.loc[common_ids].sort_index()
        cnv_df = cnv_df.loc[common_ids].sort_index()
        lab_df = lab_df.loc[common_ids].sort_index()

        # Labels: take first label column
        labels = _normalize_labels(lab_df.iloc[:, 0])

        self._mut_df = mut_df
        self._cnv_df = cnv_df
        self._labels = labels
        self._genes = list(common_genes)
        self._patient_ids = list(mut_df.index)

        logging.info("BRCADataReader: %d patients × %d genes; class balance %s",
                     len(self._patient_ids), len(self._genes),
                     dict(labels.value_counts()))

    # ------------------------------------------------------------------
    # feature matrix construction (gene-grouped [mut, cnv])
    # ------------------------------------------------------------------
    def _build_feature_matrix(self):
        # Interleave columns gene-by-gene: g0_mut, g0_cnv, g1_mut, g1_cnv, ...
        mut = self._mut_df.values.astype(np.float32)
        cnv = self._cnv_df.values.astype(np.float32)
        n_patients, n_genes = mut.shape
        x = np.empty((n_patients, n_genes * 2), dtype=np.float32)
        x[:, 0::2] = mut
        x[:, 1::2] = cnv

        # MultiIndex columns: level 0 = gene, level 1 = feature type
        cols = pd.MultiIndex.from_product(
            [self._genes, ["mut", "cnv"]],
            names=["gene", "feature"],
        )
        assert x.shape[1] == len(cols), "feature count mismatch"

        self.x = x
        self.y = self._labels.values.astype(int)
        self.info = np.asarray(self._patient_ids)
        self.columns = cols

    # ------------------------------------------------------------------
    # stratified train/val/test split (single split)
    # ------------------------------------------------------------------
    def _stratified_split(self):
        # First split off the test set
        idx = np.arange(len(self.y))
        idx_train_val, idx_test = train_test_split(
            idx, test_size=self.test_size, stratify=self.y,
            random_state=self.random_state,
        )
        # Then split train/val from the remaining 90%
        idx_train, idx_val = train_test_split(
            idx_train_val,
            test_size=self.val_size,
            stratify=self.y[idx_train_val],
            random_state=self.random_state,
        )

        # Z-score CNV using TRAIN stats only (per-column).
        if self.zscore_cnv:
            cnv_cols = np.arange(1, self.x.shape[1], 2)  # odd columns are CNV
            mu = self.x[idx_train][:, cnv_cols].mean(axis=0, keepdims=True)
            sd = self.x[idx_train][:, cnv_cols].std(axis=0, keepdims=True) + 1e-8
            for split in (idx_train, idx_val, idx_test):
                self.x[split[:, None], cnv_cols] = (
                    self.x[split[:, None], cnv_cols] - mu
                ) / sd

        # Cache the splits
        self._splits = (idx_train, idx_val, idx_test)

        logging.info("BRCADataReader split: train=%d val=%d test=%d",
                     len(idx_train), len(idx_val), len(idx_test))

    # ------------------------------------------------------------------
    # Public interface — matches ProstateDataPaper
    # ------------------------------------------------------------------
    def get_train_validate_test(self):
        idx_train, idx_val, idx_test = self._splits
        return (
            self.x[idx_train], self.x[idx_val], self.x[idx_test],
            self.y[idx_train], self.y[idx_val], self.y[idx_test],
            self.info[idx_train], self.info[idx_val], self.info[idx_test],
            self.columns,
        )
