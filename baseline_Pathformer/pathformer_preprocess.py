"""
Build Pathformer-format inputs from a (cnv, mutation, labels, pathways) dataset.

Outputs written to --output_dir:
    gene_all.txt                            (all genes in input CNV/mut, 1 per line)
    gene_select.txt                         (genes that appear in any pathway)
    modal_type_all.txt                      ("CNV\nmutation")
    pathway_gene_w.npy                      (N_select_genes x N_pathways, 0/1 mask)
    pathway_crosstalk_network.npy           (N_pathways x N_pathways)
    data_all.npy                            (N_samples x N_genes_all x 2) float32
    sample_cross.tsv                        (index, id, y, dataset_1_new ... dataset_K_new)
                                            Stratified K-fold CV. Each fold's held-out
                                            block is `test` (1/K of total); from the
                                            remaining trainval pool, 10% is taken
                                            (stratified) as `validation` and the rest
                                            is `train`. With K=5 this yields
                                            72/8/20 train/val/test per fold.
                                            Random state = --seed.

The TSV index column is what Pathformer_main reads via label['dataset_<i>_new'].
Sample ordering in data_all.npy matches the TSV row order.
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def build_pathway_mask(gene_select, pathway_df):
    gene_idx = {g: i for i, g in enumerate(gene_select)}
    G = len(gene_select)
    P = len(pathway_df)
    mask = np.zeros((G, P), dtype=np.float32)
    for j, genes_str in enumerate(pathway_df["Genes"].fillna("").values):
        for g in genes_str.split(","):
            g = g.strip()
            if g in gene_idx:
                mask[gene_idx[g], j] = 1.0
    return mask


def make_kfold_splits(y, n_splits, seed):
    """Return list of length n_splits, each entry an array of {'train','validation','test'} per sample.

    Outer loop: stratified K-fold gives a 1/K test set for each fold.
    Inner step: stratified split takes 10% of the remaining trainval pool for
    validation, leaving 90% for training. With K=5 this yields 72/8/20.
    """
    n = len(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = []
    for fold, (tr_val_idx, test_idx) in enumerate(skf.split(np.zeros(n), y)):
        y_tv = y[tr_val_idx]
        tr_idx, val_idx = train_test_split(
            tr_val_idx, test_size=0.1, stratify=y_tv, random_state=seed + fold,
        )
        split = np.array(["train"] * n, dtype=object)
        split[val_idx] = "validation"
        split[test_idx] = "test"
        splits.append(split)
    return splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnv_file", required=True)
    ap.add_argument("--mutation_file", required=True)
    ap.add_argument("--label_file", required=True,
                    help="CSV with columns id,response (response is 0/1 target).")
    ap.add_argument("--pathway_file", required=True,
                    help="CSV with columns Pathway_ID,Pathway_Name,Genes (comma sep gene symbols).")
    ap.add_argument("--adjacency_file", required=True,
                    help="CSV: pathway-pathway adjacency, first col = Pathway_ID, cols = Pathway_IDs.")
    ap.add_argument("--label_col", default="response")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_folds", type=int, default=5,
                    help="Number of stratified CV folds. Per fold, test = 1/K of "
                         "samples; from the trainval remainder, 10%% is taken as "
                         "validation. K=5 gives 72/8/20 train/val/test.")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading CNV...")
    cnv = pd.read_csv(args.cnv_file, index_col=0)
    print(f"  shape={cnv.shape}")

    print("Loading mutation...")
    mut = pd.read_csv(args.mutation_file, index_col=0)
    print(f"  shape={mut.shape}")

    print("Loading labels...")
    lab = pd.read_csv(args.label_file)
    id_col = lab.columns[0]
    print(f"  shape={lab.shape}  id_col={id_col}  label_col={args.label_col}")

    print("Loading pathways + adjacency...")
    pw = pd.read_csv(args.pathway_file)
    adj = pd.read_csv(args.adjacency_file, index_col=0)
    assert list(pw["Pathway_ID"]) == list(adj.index) == list(adj.columns), \
        "pathway file and adjacency file must list the same Pathway_IDs in the same order"
    print(f"  pathways={len(pw)}  adjacency={adj.shape}")

    # Align gene columns between CNV and mutation.
    assert list(cnv.columns) == list(mut.columns), \
        "CNV and mutation must have identical gene columns (same order)"
    genes_all = list(cnv.columns)
    print(f"genes (all)={len(genes_all)}")

    # Common patients across all three tables.
    common_ids = sorted(
        set(cnv.index.astype(str)) & set(mut.index.astype(str)) & set(lab[id_col].astype(str))
    )
    print(f"common patients={len(common_ids)}")
    assert len(common_ids) > 10, "Too few common patients."

    cnv = cnv.loc[common_ids]
    mut = mut.loc[common_ids]
    lab = lab.set_index(id_col).loc[common_ids].reset_index()

    # Genes that appear in at least one pathway.
    pw_genes = set()
    for g in pw["Genes"].fillna(""):
        for s in g.split(","):
            s = s.strip()
            if s:
                pw_genes.add(s)
    gene_select = [g for g in genes_all if g in pw_genes]
    print(f"genes (in any pathway)={len(gene_select)}")
    assert len(gene_select) > 100, "Pathway/gene overlap is suspiciously small."

    # K-fold stratified CV; per fold, test = 1/K, val = 10% of trainval.
    y = lab[args.label_col].astype(int).values
    splits = make_kfold_splits(y, args.n_folds, args.seed)
    for i, sp in enumerate(splits, start=1):
        n_tr = (sp == "train").sum()
        n_val = (sp == "validation").sum()
        n_te = (sp == "test").sum()
        print(f"  fold {i}: train={n_tr}  val={n_val}  test={n_te}")

    # Build pathway mask (G_select x P).
    print("Building pathway gene mask...")
    mask = build_pathway_mask(gene_select, pw)
    print(f"  mask shape={mask.shape}  density={mask.mean():.4f}")

    # Stack CNV + mutation into (N, G, 2) float32.
    print("Stacking modalities into (N, G, 2)...")
    data = np.stack(
        [cnv.values.astype(np.float32), mut.values.astype(np.float32)], axis=-1
    )
    print(f"  data shape={data.shape}")

    # Pathway crosstalk matrix.
    cross = adj.values.astype(np.float32)
    cross[np.isnan(cross)] = 0.0

    # Write outputs.
    out = args.output_dir
    with open(os.path.join(out, "gene_all.txt"), "w") as f:
        f.write("\n".join(genes_all) + "\n")
    with open(os.path.join(out, "gene_select.txt"), "w") as f:
        f.write("\n".join(gene_select) + "\n")
    with open(os.path.join(out, "modal_type_all.txt"), "w") as f:
        f.write("CNV\nmutation\n")

    np.save(os.path.join(out, "pathway_gene_w.npy"), mask)
    np.save(os.path.join(out, "pathway_crosstalk_network.npy"), cross)
    np.save(os.path.join(out, "data_all.npy"), data)

    label_tsv = pd.DataFrame({"id": common_ids, "y": y})
    for i, sp in enumerate(splits, start=1):
        label_tsv[f"dataset_{i}_new"] = sp
    label_tsv.to_csv(os.path.join(out, "sample_cross.tsv"), sep="\t", index=True)

    print("Done. Outputs in", out)


if __name__ == "__main__":
    main()
