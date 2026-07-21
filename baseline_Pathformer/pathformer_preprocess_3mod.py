"""Build Pathformer-format inputs for the 3-modality BRCA PAM50 task.

Uses the shared data contract (`brca_pam50_data`) so the patients, integer
labels, and 5-fold splits match every other revised baseline. Stacks CNV,
mutation, expression into (N, G, 3) float32 and writes the same artifact set as
the original 2-modality preprocess.

Outputs written to --output_dir:
    gene_all.txt                    (all genes, 1 per line)
    gene_select.txt                 (genes that appear in any pathway)
    modal_type_all.txt              ("CNV\nmutation\nexpression")
    pathway_gene_w.npy              (N_select_genes x N_pathways, 0/1 mask)
    pathway_crosstalk_network.npy   (N_pathways x N_pathways)
    data_all.npy                    (N_samples x N_genes_all x 3) float32
    sample_cross.tsv                (index, id, y, dataset_1_new ... dataset_K_new)

Modality stack order is CNV, mutation, expression and modal_type_all.txt
matches it line-for-line. Fold assignments come from D.make_folds(y), so the
exact same (train/val/test) splits are used as the shared contract.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

import brca_pam50_data as D  # noqa: E402

PATHWAY_FILE = "/lustre/hdd/LAS/weile-lab/howlader/Graph_Transformer/data_tcga/brca/filtered_pathways.csv"
ADJACENCY_FILE = "/lustre/hdd/LAS/weile-lab/howlader/Graph_Transformer/data_tcga/brca/adjacency_matrix.csv"

# Stack order; modal_type_all.txt is written to match exactly.
MODAL_STACK = ["cnv", "mutation", "expression"]
MODAL_LABELS = ["CNV", "mutation", "expression"]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pathway_file", default=PATHWAY_FILE)
    ap.add_argument("--adjacency_file", default=ADJACENCY_FILE)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading modalities via brca_pam50_data...")
    mods, y, patients, genes = D.load_modalities()
    for name in MODAL_STACK:
        assert name in mods, f"missing modality {name}"
        assert mods[name].shape == (len(patients), len(genes)), \
            f"{name} shape {mods[name].shape} != ({len(patients)},{len(genes)})"
    print(f"  patients={len(patients)}  genes={len(genes)}  classes={int(y.max()) + 1}")

    print("Loading pathways + adjacency...")
    pw = pd.read_csv(args.pathway_file)
    adj = pd.read_csv(args.adjacency_file, index_col=0)
    assert list(pw["Pathway_ID"]) == list(adj.index) == list(adj.columns), \
        "pathway file and adjacency file must list the same Pathway_IDs in the same order"
    print(f"  pathways={len(pw)}  adjacency={adj.shape}")

    genes_all = list(genes)

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

    # Use the shared 5-fold splits (seed 42, val = 10% of trainval).
    folds = D.make_folds(y)
    n = len(y)
    splits = []
    for fold_i, (tr_idx, val_idx, test_idx) in enumerate(folds, start=1):
        sp = np.array(["train"] * n, dtype=object)
        sp[val_idx] = "validation"
        sp[test_idx] = "test"
        splits.append(sp)
        print(f"  fold {fold_i}: train={(sp == 'train').sum()}  "
              f"val={(sp == 'validation').sum()}  test={(sp == 'test').sum()}")

    print("Building pathway gene mask...")
    mask = build_pathway_mask(gene_select, pw)
    print(f"  mask shape={mask.shape}  density={mask.mean():.4f}")

    print("Stacking modalities into (N, G, 3)...")
    data = np.stack([mods[name].astype(np.float32) for name in MODAL_STACK], axis=-1)
    print(f"  data shape={data.shape}  (order: {MODAL_LABELS})")

    cross = adj.values.astype(np.float32)
    cross[np.isnan(cross)] = 0.0

    out = args.output_dir
    with open(os.path.join(out, "gene_all.txt"), "w") as f:
        f.write("\n".join(genes_all) + "\n")
    with open(os.path.join(out, "gene_select.txt"), "w") as f:
        f.write("\n".join(gene_select) + "\n")
    with open(os.path.join(out, "modal_type_all.txt"), "w") as f:
        f.write("\n".join(MODAL_LABELS) + "\n")

    np.save(os.path.join(out, "pathway_gene_w.npy"), mask)
    np.save(os.path.join(out, "pathway_crosstalk_network.npy"), cross)
    np.save(os.path.join(out, "data_all.npy"), data)

    label_tsv = pd.DataFrame({"id": [str(p) for p in patients], "y": y.astype(int)})
    for i, sp in enumerate(splits, start=1):
        label_tsv[f"dataset_{i}_new"] = sp
    label_tsv.to_csv(os.path.join(out, "sample_cross.tsv"), sep="\t", index=True)

    print("Done. Outputs in", out)


if __name__ == "__main__":
    main()
