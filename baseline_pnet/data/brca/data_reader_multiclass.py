"""
Multi-class variant of BRCADataReader.

Same interface, same on-disk layout (mutation_data.csv, cnv_data.csv,
<labels_filename>), but accepts integer labels in `{0, 1, ..., K-1}` and
carries `num_classes`. Used for PAM50 (5-class), KIRC subtype (4-class),
cancer-type-of-origin (32-class), etc.

Labels are taken from the first non-id column of the labels CSV (already
integer-encoded — `0..K-1`). No 0/1 coercion; no Primary/Metastatic mapping.
"""
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _coerce_multiclass_labels(label_series: pd.Series, num_classes: Optional[int]) -> np.ndarray:
    y = pd.to_numeric(label_series, errors="raise").astype(int).values
    if (y < 0).any():
        raise ValueError("multiclass labels must be non-negative integers; got "
                         f"min={y.min()}")
    if num_classes is not None and y.max() >= num_classes:
        raise ValueError(f"label value {y.max()} exceeds num_classes={num_classes}")
    return y


class BRCAMulticlassDataReader:
    """
    Multi-class 2-modality (mutation + CNV) reader.

    Parameters mirror BRCADataReader, plus:
        num_classes : int
            Number of distinct classes. If None, inferred from labels.
    """

    def __init__(self,
                 data_dir: str,
                 labels_filename: str = "patient_labels.csv",
                 num_classes: Optional[int] = None,
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
        self.num_classes = num_classes
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

        mut_df = mut_df.set_index(mut_df.columns[0])
        cnv_df = cnv_df.set_index(cnv_df.columns[0])
        lab_df = lab_df.set_index(lab_df.columns[0])
        for df in (mut_df, cnv_df, lab_df):
            df.index = df.index.astype(str)

        if self.selected_genes_filename is not None:
            sel = pd.read_csv(self.selected_genes_filename, header=None)[0].astype(str).tolist()
            mut_df = mut_df[[g for g in sel if g in mut_df.columns]]
            cnv_df = cnv_df[[g for g in sel if g in cnv_df.columns]]
            logging.info("BRCAMulticlassDataReader: restricted to %d selected genes.", len(sel))

        common_genes = [g for g in mut_df.columns if g in cnv_df.columns]
        if not common_genes:
            raise ValueError("No genes overlap between mutation and CNV tables.")
        mut_df = mut_df[common_genes]
        cnv_df = cnv_df[common_genes]

        common_ids = (mut_df.index
                      .intersection(cnv_df.index)
                      .intersection(lab_df.index))
        if len(common_ids) == 0:
            raise ValueError("No patient IDs overlap between modalities and labels.")
        mut_df = mut_df.loc[common_ids].sort_index()
        cnv_df = cnv_df.loc[common_ids].sort_index()
        lab_df = lab_df.loc[common_ids].sort_index()

        labels = _coerce_multiclass_labels(lab_df.iloc[:, 0], self.num_classes)
        if self.num_classes is None:
            self.num_classes = int(labels.max()) + 1

        self._mut_df = mut_df
        self._cnv_df = cnv_df
        self._labels = labels
        self._genes = list(common_genes)
        self._patient_ids = list(mut_df.index)

        class_counts = {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}
        logging.info("BRCAMulticlassDataReader: %d patients × %d genes × %d classes; balance %s",
                     len(self._patient_ids), len(self._genes), self.num_classes, class_counts)

    def _build_feature_matrix(self):
        mut = self._mut_df.values.astype(np.float32)
        cnv = self._cnv_df.values.astype(np.float32)
        n_patients, n_genes = mut.shape
        x = np.empty((n_patients, n_genes * 2), dtype=np.float32)
        x[:, 0::2] = mut
        x[:, 1::2] = cnv

        cols = pd.MultiIndex.from_product(
            [self._genes, ["mut", "cnv"]],
            names=["gene", "feature"],
        )
        assert x.shape[1] == len(cols), "feature count mismatch"

        self.x = x
        self.y = self._labels.astype(int)
        self.info = np.asarray(self._patient_ids)
        self.columns = cols

    def _stratified_split(self):
        idx = np.arange(len(self.y))
        idx_train_val, idx_test = train_test_split(
            idx, test_size=self.test_size, stratify=self.y,
            random_state=self.random_state,
        )
        idx_train, idx_val = train_test_split(
            idx_train_val,
            test_size=self.val_size,
            stratify=self.y[idx_train_val],
            random_state=self.random_state,
        )

        if self.zscore_cnv:
            cnv_cols = np.arange(1, self.x.shape[1], 2)
            mu = self.x[idx_train][:, cnv_cols].mean(axis=0, keepdims=True)
            sd = self.x[idx_train][:, cnv_cols].std(axis=0, keepdims=True) + 1e-8
            for split in (idx_train, idx_val, idx_test):
                self.x[split[:, None], cnv_cols] = (
                    self.x[split[:, None], cnv_cols] - mu
                ) / sd

        self._splits = (idx_train, idx_val, idx_test)

        logging.info("BRCAMulticlassDataReader split: train=%d val=%d test=%d",
                     len(idx_train), len(idx_val), len(idx_test))

    def get_train_validate_test(self):
        idx_train, idx_val, idx_test = self._splits
        return (
            self.x[idx_train], self.x[idx_val], self.x[idx_test],
            self.y[idx_train], self.y[idx_val], self.y[idx_test],
            self.info[idx_train], self.info[idx_val], self.info[idx_test],
            self.columns,
        )
