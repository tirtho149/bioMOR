"""
Train Pathformer on a single fold of a 2-modality (CNV + mutation) dataset.

This script wraps the *unmodified* `pathformer_model` from
Pathformer_code/Pathformer.py. The original Pathformer_main.py auto-enables an
embedding layer when N_PRJ <= 2, but with N_PRJ == 2 its dim formula collapses
(embeding_num/row_dim == 1) and the axial attention fails on a row dim of 2.
We sidestep that by using the no-embedding code path the original uses for
N_PRJ > 2 (embeding=False, row_dim=N_PRJ). The model class itself is untouched.

Hyperparameters are aligned with SKILLS.md (the cross-baseline config used by
Graph_Transformer/multimodal/main_soft_masking_2modal.py).
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from einops import repeat
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "Pathformer_code"))
from Pathformer import pathformer_model  # noqa: E402
from utils import SCDataset, setup_seed  # noqa: E402


def load_inputs(data_dir, fold):
    label = pd.read_csv(os.path.join(data_dir, "sample_cross.tsv"), sep="\t", index_col=0)
    data = np.load(os.path.join(data_dir, "data_all.npy"))
    mask = np.load(os.path.join(data_dir, "pathway_gene_w.npy"))
    cross = np.load(os.path.join(data_dir, "pathway_crosstalk_network.npy"))
    cross[np.isnan(cross)] = 0.0

    gene_all = pd.read_csv(os.path.join(data_dir, "gene_all.txt"), header=None)
    gene_all.columns = ["gene_id"]
    gene_all["index"] = range(len(gene_all))
    gene_all = gene_all.set_index("gene_id")
    gene_select = pd.read_csv(os.path.join(data_dir, "gene_select.txt"), header=None)
    gene_select_index = list(gene_all.loc[list(gene_select[0]), "index"])

    modal_all = pd.read_csv(os.path.join(data_dir, "modal_type_all.txt"), header=None)
    modal_select_index = list(range(len(modal_all)))

    fold_col = f"dataset_{fold}_new"
    train_idx = list(label.index[label[fold_col] == "train"])
    val_idx = list(label.index[label[fold_col] == "validation"])
    test_idx = list(label.index[label[fold_col] == "test"])

    def slice_data(rows):
        return data[rows][:, gene_select_index][:, :, modal_select_index]

    def y_for(rows):
        return label.loc[rows, ["y"]].values.astype(int)

    parts = {
        "train": (slice_data(train_idx), y_for(train_idx)),
        "val":   (slice_data(val_idx),   y_for(val_idx)),
        "test":  (slice_data(test_idx),  y_for(test_idx)),
    }

    sample_ids = list(label.loc[test_idx, "id"])
    return parts, mask, cross, sample_ids, train_idx


def evaluate(model, loader, pathway_network, device, n_classes):
    model.eval()
    ys, probs = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)[:, 0]
            edges = repeat(pathway_network, "i j -> b i j", b=x.shape[0])
            logits = model(edges, x.permute(0, 2, 1), output_attentions=False)
            ys.append(y.cpu().numpy())
            probs.append(logits.cpu().numpy())
    y_true = np.concatenate(ys)
    p = np.concatenate(probs)
    y_pred = p.argmax(axis=1)
    metrics = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }
    if n_classes == 2:
        metrics["f1_binary"] = float(f1_score(y_true, y_pred, average="binary", pos_label=1))
        metrics["auc"] = float(roc_auc_score(y_true, p[:, 1]))
    else:
        metrics["auc"] = float(roc_auc_score(y_true, p, multi_class="ovr", average="macro"))
    return metrics, y_true, p


def train_one_fold(args):
    setup_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parts, mask_np, cross_np, test_ids, train_idx = load_inputs(args.data_dir, args.fold)
    (Xtr, ytr), (Xval, yval), (Xte, yte) = parts["train"], parts["val"], parts["test"]
    print(f"fold {args.fold}  train={Xtr.shape}  val={Xval.shape}  test={Xte.shape}")

    train_loader = DataLoader(SCDataset(Xtr, ytr), batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(SCDataset(Xval, yval), batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(SCDataset(Xte, yte), batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    mask = torch.LongTensor(mask_np.astype(np.int64))
    pathway_network = torch.Tensor(cross_np).to(device)

    n_modalities = Xtr.shape[2]   # 2 here
    n_pathways = mask_np.shape[1]
    n_classes = int(np.unique(np.concatenate([ytr, yval, yte])).size)

    # The no-embedding branch from Pathformer_main.py (used when N_PRJ > 2).
    # We force it for N_PRJ == 2 because the embedding branch has a dim bug for that case.
    embeding = False
    row_dim = n_modalities
    classifier_input = n_modalities * n_pathways

    model = pathformer_model(
        mask_raw=mask,
        row_dim=row_dim,
        col_dim=n_pathways,
        depth=args.depth,
        heads=args.heads,
        dim_head=args.dim_head,
        classifier_input=classifier_input,
        classifier_dim=[300, 200, 100],
        label_dim=n_classes,
        embeding=embeding,
        embeding_num=32,
        beta=args.beta,
        attn_dropout=args.dropout,
        ff_dropout=args.dropout,
        classifier_dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    cw = compute_class_weight("balanced", classes=np.unique(ytr[:, 0]), y=ytr[:, 0])
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(device))

    best_val_auc, best_state, best_epoch = -1.0, None, -1
    patience_left = args.patience
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running = 0.0
        seen = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)[:, 0]
            edges = repeat(pathway_network, "i j -> b i j", b=x.shape[0])
            logits = model(edges, x.permute(0, 2, 1), output_attentions=False)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1e6)
            optimizer.step()
            running += loss.item() * x.shape[0]
            seen += x.shape[0]
        tr_loss = running / max(seen, 1)

        val_metrics, _, _ = evaluate(model, val_loader, pathway_network, device, n_classes)
        dt = time.time() - t0
        print(f"  epoch {epoch:3d}  loss={tr_loss:.4f}  "
              f"val_auc={val_metrics['auc']:.4f}  val_f1m={val_metrics['f1_macro']:.4f}  "
              f"({dt:.1f}s)")
        history.append({"epoch": epoch, "train_loss": tr_loss, **{f"val_{k}": v for k, v in val_metrics.items()}})

        improved = val_metrics["auc"] > best_val_auc
        if improved:
            best_val_auc = val_metrics["auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_left = args.patience
        else:
            patience_left -= 1
            if epoch >= args.min_epochs and patience_left <= 0:
                print(f"  early stop at epoch {epoch} (best epoch {best_epoch})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics, y_true, p_test = evaluate(model, test_loader, pathway_network, device, n_classes)
    print(f"fold {args.fold} TEST  acc={test_metrics['acc']:.4f}  "
          f"auc={test_metrics['auc']:.4f}  f1_macro={test_metrics['f1_macro']:.4f}  "
          f"f1_weighted={test_metrics['f1_weighted']:.4f}")

    os.makedirs(args.save_dir, exist_ok=True)
    out = {
        "fold": args.fold,
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "test_metrics": test_metrics,
        "args": vars(args),
        "history": history,
    }
    with open(os.path.join(args.save_dir, f"fold{args.fold}_result.json"), "w") as f:
        json.dump(out, f, indent=2)

    if n_classes == 2:
        y_score = p_test[:, 1]
    else:
        y_score = p_test.tolist()
    preds_df = pd.DataFrame({
        "fold": args.fold,
        "sample": test_ids,
        "y_label": y_true,
        "y_score": y_score if n_classes == 2 else [json.dumps(s) for s in y_score],
    })
    preds_df.to_csv(os.path.join(args.save_dir, f"fold{args.fold}_test_predictions.csv"), index=False)

    if args.save_model:
        torch.save(model.state_dict(), os.path.join(args.save_dir, f"fold{args.fold}_best.pt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Output dir of pathformer_preprocess.py")
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--fold", type=int, required=True, help="1-indexed fold number")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--min_epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dim_head", type=int, default=32)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_model", action="store_true")
    args = ap.parse_args()
    train_one_fold(args)


if __name__ == "__main__":
    main()
