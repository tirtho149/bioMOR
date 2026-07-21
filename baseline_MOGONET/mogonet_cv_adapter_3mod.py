"""
MOGONET 3-VIEW Data Adapter for the TCGA BRCA PAM50 (5-class) task.

NEW file (does not modify the original 2-view mogonet_cv_adapter.py).

Uses the shared data contract at repo root:
    import brca_pam50_data as D
    mods, y, patients, genes = D.load_modalities()   # 518 patients, 5 classes
    folds = D.make_folds(y)                           # 5 (train,val,test) tuples

View mapping (per task spec):
    view1 = cnv         (continuous -> StandardScaler then min-max to [0,1])
    view2 = expression  (continuous -> StandardScaler then min-max to [0,1])
    view3 = mutation    (binarized to {0,1}, threshold 0)

This mirrors the ORIGINAL adapter's per-view preprocessing:
    - continuous omics: StandardScaler + per-feature min-max scaling
    - mutation: binarize ( > threshold -> 1 )
All scalers / min-max ranges are FIT ON THE TRAIN ROWS ONLY (train_idx, the
val rows are excluded from the fit, matching "fit on train only"), then applied
to train / val / test rows of the same fold.

Writes, for each fold, into <output_dir>/fold_<k>/:
    1_tr.csv 2_tr.csv 3_tr.csv  + labels_tr.csv samples_tr.csv
    1_val.csv 2_val.csv 3_val.csv + labels_val.csv samples_val.csv
    1_te.csv 2_te.csv 3_te.csv  + labels_te.csv samples_te.csv
    {1,2,3}_featname.csv
These are exactly the file names train_test_cv.prepare_trte_data expects, with
an extra third view.
"""

import os
import sys
import argparse

import numpy as np
from sklearn.preprocessing import StandardScaler

# Make the repo-root shared helpers importable regardless of cwd.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import brca_pam50_data as D  # noqa: E402

# view index (1-based) -> shared modality name
VIEW_TO_MODALITY = {1: "cnv", 2: "expression", 3: "mutation"}
CONTINUOUS_VIEWS = {1, 2}   # scaled like the original CNV view
BINARY_VIEWS = {3}          # binarized like the original mutation view


def _scale_continuous(train_rows, *other_row_blocks):
    """StandardScaler + per-feature min-max to [0,1], FIT ON train_rows only.

    Returns scaled (train, *others) in the same order given.
    """
    scaler = StandardScaler()
    tr_std = scaler.fit_transform(train_rows)

    col_min = tr_std.min(axis=0)
    col_max = tr_std.max(axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1.0  # avoid divide-by-zero on constant features

    def _apply(block):
        std = scaler.transform(block)
        return (std - col_min) / col_range

    return [(tr_std - col_min) / col_range] + [_apply(b) for b in other_row_blocks]


def _binarize(block, threshold=0):
    return (block > threshold).astype(np.float32)


def build(output_dir, mutation_threshold=0):
    print("Loading shared modalities via brca_pam50_data ...")
    mods, y, patients, genes = D.load_modalities()
    patients = np.asarray(patients)
    y = np.asarray(y).astype(int)
    print(f"  patients={len(patients)}  genes={len(genes)}  classes={len(np.unique(y))}")
    print(f"  label distribution: {np.bincount(y)}")

    folds = D.make_folds(y)
    os.makedirs(output_dir, exist_ok=True)

    feat_names = list(map(str, genes))

    for k, (train_idx, val_idx, test_idx) in enumerate(folds, start=1):
        fold_dir = os.path.join(output_dir, f"fold_{k}")
        os.makedirs(fold_dir, exist_ok=True)

        for view in (1, 2, 3):
            name = VIEW_TO_MODALITY[view]
            X = mods[name]
            Xtr, Xval, Xte = X[train_idx], X[val_idx], X[test_idx]

            if view in CONTINUOUS_VIEWS:
                # Fit scaler/min-max on TRAIN rows only, apply to all splits.
                Xtr, Xval, Xte = _scale_continuous(Xtr, Xval, Xte)
            else:  # binary mutation view
                Xtr = _binarize(Xtr, mutation_threshold)
                Xval = _binarize(Xval, mutation_threshold)
                Xte = _binarize(Xte, mutation_threshold)

            np.savetxt(os.path.join(fold_dir, f"{view}_tr.csv"), Xtr, delimiter=',')
            np.savetxt(os.path.join(fold_dir, f"{view}_val.csv"), Xval, delimiter=',')
            np.savetxt(os.path.join(fold_dir, f"{view}_te.csv"), Xte, delimiter=',')
            with open(os.path.join(fold_dir, f"{view}_featname.csv"), 'w') as f:
                for nm in feat_names:
                    f.write(f"{nm}\n")

        # labels + sample ids per split
        np.savetxt(os.path.join(fold_dir, "labels_tr.csv"), y[train_idx], delimiter=',')
        np.savetxt(os.path.join(fold_dir, "labels_val.csv"), y[val_idx], delimiter=',')
        np.savetxt(os.path.join(fold_dir, "labels_te.csv"), y[test_idx], delimiter=',')
        np.savetxt(os.path.join(fold_dir, "samples_tr.csv"), patients[train_idx], fmt='%s', delimiter=',')
        np.savetxt(os.path.join(fold_dir, "samples_val.csv"), patients[val_idx], fmt='%s', delimiter=',')
        np.savetxt(os.path.join(fold_dir, "samples_te.csv"), patients[test_idx], fmt='%s', delimiter=',')

        print(f"Fold {k}: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
              f"-> {fold_dir}")

    print("\n3-view MOGONET fold data written to:", output_dir)
    print("Views: 1=cnv  2=expression  3=mutation | classes: 5")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Build 3-view MOGONET inputs for BRCA PAM50")
    parser.add_argument("--output_dir",
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "brca_pam50_3mod_adapted"),
                        help="Output base dir containing fold_1..5")
    parser.add_argument("--mutation_threshold", type=float, default=0)
    args = parser.parse_args()
    build(args.output_dir, mutation_threshold=args.mutation_threshold)


if __name__ == "__main__":
    main()
