"""Pathformer for 2-modality pan-cancer Cox survival prediction.

Self-contained survival revision of the Pathformer baseline. Builds the
Pathformer inputs in memory from the shared survival contract (CNV + mutation
stacked into (N, G, 2)), reuses the *unmodified* pathformer_model with
label_dim=1 (single log-relative-hazard, embeding=False / row_dim=2 like the
2-modal classification path), and trains with the Cox partial log-likelihood.
Metric: Harrell's C-index. 5-fold stratified splits.

Run in the `path` conda env, with this dir on PYTHONPATH:
    python pathformer_survival.py     (SMOKE=1 for a fast 1-fold smoke test)
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
from einops import repeat
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "baseline_Pathformer", "Pathformer_code"))

import pan_survival_data as D
from survival_metrics import cox_partial_loglik_loss, harrell_c_index, fold_metrics, write_metrics_json, write_risk_scores
from Pathformer import pathformer_model
from utils import setup_seed

PATHWAY_FILE = os.path.join(D.DATA_DIR, "filtered_pathways.csv")
ADJACENCY_FILE = os.path.join(D.DATA_DIR, "adjacency_matrix.csv")
RESULTS_JSON = os.path.join(HERE, os.environ.get("SURV_RESULTS_DIR", "results"), "Pathformer_survival_metrics.json")
RISK_CSV = os.path.join(HERE, os.environ.get("SURV_RESULTS_DIR", "results"), "Pathformer_risk_scores.csv")
SMOKE = os.environ.get("SMOKE", "0") == "1"

MODAL_STACK = D.MODALITY_NAMES  # 2- or 3-modal depending on SURV_3MODAL


class SurvDataset(Dataset):
    def __init__(self, X, time, event):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.time = torch.tensor(time, dtype=torch.float32)
        self.event = torch.tensor(event, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.time[i], self.event[i]


def build_mask(gene_select, pathway_df):
    gene_idx = {g: i for i, g in enumerate(gene_select)}
    mask = np.zeros((len(gene_select), len(pathway_df)), dtype=np.float32)
    for j, genes_str in enumerate(pathway_df["Genes"].fillna("").values):
        for g in genes_str.split(","):
            g = g.strip()
            if g in gene_idx:
                mask[gene_idx[g], j] = 1.0
    return mask


@torch.no_grad()
def predict(model, loader, pathway_network, device):
    model.eval()
    risks, times, events = [], [], []
    for x, t, e in loader:
        x = x.to(device)
        edges = repeat(pathway_network, "i j -> b i j", b=x.shape[0])
        risk = model(edges, x.permute(0, 2, 1), output_attentions=False).reshape(-1)
        risks.append(risk.cpu().numpy()); times.append(t.numpy()); events.append(e.numpy())
    return np.concatenate(risks), np.concatenate(times), np.concatenate(events)


def main():
    setup_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading 2-modality survival data...")
    mods, time, event, patients, genes = D.load_modalities()
    print(f"patients={len(patients)} genes={len(genes)} events={int(event.sum())}")

    pw = pd.read_csv(PATHWAY_FILE)
    adj = pd.read_csv(ADJACENCY_FILE, index_col=0)
    assert list(pw["Pathway_ID"]) == list(adj.index) == list(adj.columns)
    cross = adj.values.astype(np.float32)
    cross[np.isnan(cross)] = 0.0

    pw_genes = set()
    for g in pw["Genes"].fillna(""):
        pw_genes.update(s.strip() for s in g.split(",") if s.strip())
    gene_select_index = [i for i, g in enumerate(genes) if g in pw_genes]
    gene_select = [genes[i] for i in gene_select_index]
    print(f"genes in any pathway: {len(gene_select)}")

    mask_np = build_mask(gene_select, pw)
    n_pathways = mask_np.shape[1]
    # Standardize continuous modalities (expression/cnv) so no single modality
    # dominates by scale; mutation is left binary. (Global stats: Pathformer
    # stacks once outside the fold loop.)
    def _std(a):
        m = a.mean(0, keepdims=True); s = a.std(0, keepdims=True); s[s == 0] = 1.0
        return ((a - m) / s).astype(np.float32)
    layers = []
    for name in MODAL_STACK:
        a = mods[name].astype(np.float32)
        layers.append(a if name == "mutation" else _std(a))
    data = np.stack(layers, axis=-1)
    data = data[:, gene_select_index, :]   # (N, n_select_genes, n_modalities)
    print(f"data shape={data.shape}  mask={mask_np.shape}")

    mask = torch.LongTensor(mask_np.astype(np.int64))
    pathway_network = torch.Tensor(cross).to(device)

    folds = D.make_folds(event)
    if SMOKE:
        folds = folds[:1]

    n_modalities = data.shape[2]
    batch_size = D.BATCH_SIZE
    epochs = D.MAX_EPOCHS
    min_epochs = D.MIN_EPOCHS
    patience = D.PATIENCE
    lr, weight_decay, dropout = D.LR, D.WEIGHT_DECAY, 0.2

    per_fold = []
    risk_records = []
    for fold_i, (tr_idx, val_idx, test_idx) in enumerate(folds, 1):
        print(f"\n{'='*60}\nFOLD {fold_i}/{len(folds)}\n{'='*60}")
        setup_seed(42 + fold_i)

        train_loader = DataLoader(SurvDataset(data[tr_idx], time[tr_idx], event[tr_idx]),
                                  batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
        val_loader = DataLoader(SurvDataset(data[val_idx], time[val_idx], event[val_idx]),
                                batch_size=batch_size, shuffle=False, pin_memory=True)
        test_loader = DataLoader(SurvDataset(data[test_idx], time[test_idx], event[test_idx]),
                                 batch_size=batch_size, shuffle=False, pin_memory=True)

        model = pathformer_model(
            mask_raw=mask, row_dim=n_modalities, col_dim=n_pathways,
            depth=3, heads=8, dim_head=32,
            classifier_input=n_modalities * n_pathways,
            classifier_dim=[300, 200, 100], label_dim=1,
            embeding=False, embeding_num=32, beta=1.0,
            attn_dropout=dropout, ff_dropout=dropout, classifier_dropout=dropout,
        ).to(device)
        # The shared Pathformer classifier ends in Softmax(dim=1); with
        # label_dim=1 that collapses every output to 1.0 (constant risk, C=0.5).
        # Replace it with Identity so the head emits a raw log-relative-hazard.
        model.classifier_model.layer[-1] = torch.nn.Identity()

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        best_val_c, best_state, no_improve = -np.inf, None, 0
        for epoch in range(1, epochs + 1):
            model.train()
            for x, t, e in train_loader:
                x, t, e = x.to(device), t.to(device), e.to(device)
                edges = repeat(pathway_network, "i j -> b i j", b=x.shape[0])
                risk = model(edges, x.permute(0, 2, 1), output_attentions=False).reshape(-1)
                loss = cox_partial_loglik_loss(risk, t, e)
                optimizer.zero_grad()
                if torch.isfinite(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    optimizer.step()
            vc = harrell_c_index(*predict(model, val_loader, pathway_network, device))
            print(f"  epoch {epoch:3d}  val_C={vc:.4f}", flush=True)
            if not np.isnan(vc) and vc > best_val_c:
                best_val_c, no_improve = vc, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1
            if epoch >= min_epochs and no_improve >= patience:
                print(f"  early stop @ epoch {epoch}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        te_risk, te_time, te_event = predict(model, test_loader, pathway_network, device)
        m = fold_metrics(te_risk, te_time, te_event)
        per_fold.append(m)
        risk_records.append({"test_idx": test_idx, "risk": te_risk,
                             "time": te_time, "event": te_event})
        print(f"  Fold {fold_i} C-index={m['c_index']:.4f} (n={m['n']}, events={m['events']})")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_metrics_json(RESULTS_JSON, "Pathformer", os.environ.get("SURV_DATASET_TAG", "pan_survival_cox"), per_fold)
    write_risk_scores(RISK_CSV, patients, risk_records)
    print(f"\nWrote metrics to {RESULTS_JSON}")
    print(f"Wrote risk scores to {RISK_CSV}")


if __name__ == "__main__":
    main()
