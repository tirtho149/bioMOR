"""
MOGONET 3-VIEW, 5-CLASS driver for the TCGA BRCA PAM50 task.

NEW file (does not modify main_mogonet_cv.py). Reuses the ORIGINAL MOGONET
model (models.py) and training primitives (train_test_cv.py / utils_adapted.py)
- the GCN/VCDN are NOT reimplemented. This script only:

  1. builds 3-view per-fold inputs via mogonet_cv_adapter_3mod.build(), which
     uses the shared splits from brca_pam50_data.make_folds (identical across
     baselines);
  2. for each fold, pretrains + trains the model with the ORIGINAL
     hyperparameters, selecting the checkpoint with the best validation
     accuracy (multiclass-safe; original used binary-F1 which is ill-defined
     here);
  3. extracts the (N_test, 5) softmax TEST probabilities at that checkpoint
     using the original test_epoch();
  4. scores them with the shared multiclass_metrics.fold_metrics and writes
     results/brca_pam50/MOGONET_metrics.json via write_metrics_json.

num_view = 3 (1=cnv, 2=expression, 3=mutation), num_class = 5.

Env vars:
  SMOKE=1            -> 1 fold, reduced epochs (quick smoke test). Default full.
  MOGONET_DATA_FOLDER -> override the adapted-data base dir.
"""

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for p in (_HERE, _REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import brca_pam50_data as D                          # noqa: E402
from multiclass_metrics import fold_metrics, write_metrics_json  # noqa: E402

import mogonet_cv_adapter_3mod as adapter            # noqa: E402
from models import init_model_dict, init_optim       # noqa: E402  (original model)
from train_test_cv import (                          # noqa: E402  (original primitives)
    prepare_trte_data, gen_trte_adj_mat, train_epoch, test_epoch, device,
)
from utils_adapted import one_hot_tensor, cal_sample_weight  # noqa: E402


def run_single_fold(fold_dir, view_list, num_class,
                    lr_e_pretrain, lr_e, lr_c,
                    num_epoch_pretrain, num_epoch,
                    adj_parameter, dim_he_list, patience, test_interval=50):
    """Train one fold; return (test_prob (N,5), y_test) at best-val checkpoint.

    Mirrors train_test_cv.train_test_single_fold but returns full softmax
    probabilities (the original only returns binary scalar metrics).
    """
    num_view = len(view_list)
    dim_hvcdn = pow(num_class, num_view)

    data_tr_list, data_trte_list, trte_idx, labels_trte = prepare_trte_data(fold_dir, view_list)

    labels_tr_tensor = torch.tensor(labels_trte[trte_idx["tr"]], dtype=torch.long, device=device)
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class).to(device)
    sample_weight_tr = torch.tensor(
        cal_sample_weight(labels_trte[trte_idx["tr"]], num_class),
        dtype=torch.float32, device=device,
    )

    adj_tr_list, adj_val_list, adj_te_list, data_val_list, data_te_list = gen_trte_adj_mat(
        data_tr_list, data_trte_list, trte_idx, adj_parameter, view_list
    )

    dim_list = [x.shape[1] for x in data_tr_list]
    print(f"  feature dims={dim_list}  vcdn_dim={dim_hvcdn}")

    model_dict = init_model_dict(num_view, num_class, dim_list, dim_he_list,
                                 dim_hvcdn, gcn_dropout=0.2, device=device)

    # Pretrain encoders (VCDN off), original schedule.
    optim_dict = init_optim(num_view, model_dict, lr_e_pretrain, lr_c)
    for epoch in range(num_epoch_pretrain):
        train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor,
                    onehot_labels_tr_tensor, sample_weight_tr,
                    model_dict, optim_dict, train_VCDN=False)

    # Full training with VCDN; early stop on validation accuracy (multiclass-safe).
    optim_dict = init_optim(num_view, model_dict, lr_e, lr_c)
    num_tr = len(trte_idx["tr"])
    val_idx_rel = list(range(num_tr, num_tr + len(trte_idx["val"])))
    te_idx_rel = list(range(num_tr, num_tr + len(trte_idx["te"])))
    y_val = labels_trte[trte_idx["val"]]
    y_te = labels_trte[trte_idx["te"]]

    best_val_acc = -1.0
    best_epoch = 0
    best_te_prob = None
    patience_counter = 0

    for epoch in range(num_epoch + 1):
        train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor,
                    onehot_labels_tr_tensor, sample_weight_tr,
                    model_dict, optim_dict, train_VCDN=True)

        if epoch % test_interval == 0:
            val_prob = test_epoch(data_val_list, adj_val_list, val_idx_rel, model_dict)
            val_acc = float((val_prob.argmax(1) == y_val).mean())
            te_prob = test_epoch(data_te_list, adj_te_list, te_idx_rel, model_dict)

            if best_te_prob is None or val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                best_te_prob = te_prob
                patience_counter = 0
            else:
                patience_counter += test_interval

            print(f"  epoch {epoch:4d}  val_acc={val_acc:.4f}  "
                  f"test_acc={(te_prob.argmax(1) == y_te).mean():.4f}"
                  + ("  <- best" if best_epoch == epoch else ""))

            if patience_counter >= patience:
                print(f"  early stop @ epoch {epoch} (best val_acc={best_val_acc:.4f} @ {best_epoch})")
                break

    return best_te_prob, y_te, best_epoch


def main():
    smoke = os.environ.get("SMOKE", "0") == "1"

    data_folder = os.environ.get(
        "MOGONET_DATA_FOLDER",
        os.path.join(_HERE, "brca_pam50_3mod_adapted"),
    )
    out_json = os.path.join(_HERE, "results", "brca_pam50", "MOGONET_metrics.json")

    view_list = [1, 2, 3]   # 1=cnv, 2=expression, 3=mutation
    num_class = 5

    # Original hyperparameters (from main_mogonet_cv.py).
    num_epoch_pretrain = 200
    num_epoch = 200
    lr_e_pretrain = 1e-4
    lr_e = 1e-4
    lr_c = 1e-4
    adj_parameter = 10
    dim_he_list = [400, 400, 200]
    patience = 25
    n_folds = 5

    if smoke:
        n_folds = 1
        num_epoch_pretrain = 5
        num_epoch = 5
        patience = 10
        print("[SMOKE] 1 fold, reduced epochs")

    print("=" * 60)
    print("MOGONET 3-VIEW 5-CLASS (BRCA PAM50)")
    print(f"device={device}  views={view_list}  num_class={num_class}")
    print(f"data_folder={data_folder}")
    print("=" * 60)

    # Build (or rebuild) the 3-view fold inputs from the shared splits.
    adapter.build(data_folder)

    per_fold = []
    for k in range(1, n_folds + 1):
        fold_dir = os.path.join(data_folder, f"fold_{k}")
        print(f"\n===== FOLD {k}/{n_folds} =====")
        te_prob, y_te, best_epoch = run_single_fold(
            fold_dir, view_list, num_class,
            lr_e_pretrain, lr_e, lr_c,
            num_epoch_pretrain, num_epoch,
            adj_parameter, dim_he_list, patience,
        )
        if te_prob is None:
            print(f"  fold {k}: no checkpoint produced; skipping")
            continue
        m = fold_metrics(y_te, te_prob, num_class)
        per_fold.append(m)
        print(f"  fold {k} metrics: acc={m['accuracy']:.4f} f1={m['f1']:.4f} "
              f"f1_macro={m['f1_macro']:.4f} auc={m['auc']:.4f} (best_epoch={best_epoch})")

    write_metrics_json(out_json, "MOGONET", "brca_pam50", num_class, per_fold)
    print(f"\nWrote metrics for {len(per_fold)} fold(s) to: {out_json}")


if __name__ == "__main__":
    main()
