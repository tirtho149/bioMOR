"""P-NET-style sparse pathway network for 2-modality pan-cancer Cox survival.

The original P-NET baseline is a Keras framework whose CV/loss/output are
class-count-coupled. For the survival suite we re-implement P-NET's defining
inductive bias -- a sparse, biologically-masked gene->pathway architecture --
as a standalone Torch model with a single log-relative-hazard (Cox) head:

  (N, G, 2 modalities)
    -> per-gene layer combining the 2 data types         (N, G)
    -> sparse gene->pathway layer (masked by pathway membership)   (N, P)
    -> dense pathway layers
    -> single risk scalar (Cox PH head)

Trained with the Cox partial log-likelihood; metric Harrell's C-index. Reuses
the same filtered_pathways gene mask as the PathCNN/Pathformer survival
baselines. Shared survival contract + 5-fold stratified splits.

    python pnet_survival.py     (SMOKE=1 for a fast 1-fold smoke test)
"""
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pan_survival_data as D
from survival_metrics import cox_partial_loglik_loss, harrell_c_index, fold_metrics, write_metrics_json, write_risk_scores

PATHWAY_FILE = os.path.join(D.DATA_DIR, "filtered_pathways.csv")
RESULTS_JSON = os.path.join(HERE, os.environ.get("SURV_RESULTS_DIR", "results"), "pnet_survival_metrics.json")
RISK_CSV = os.path.join(HERE, os.environ.get("SURV_RESULTS_DIR", "results"), "pnet_risk_scores.csv")
SMOKE = os.environ.get("SMOKE", "0") == "1"

MODAL_STACK = ["cnv", "mutation"]
SEED = D.SEED


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def build_mask(genes, pathway_df):
    """Gene->pathway membership mask (G, P), restricted to pathway genes."""
    gene_idx = {g: i for i, g in enumerate(genes)}
    cols = []
    for genes_str in pathway_df["Genes"].fillna("").values:
        col = np.zeros(len(genes), dtype=np.float32)
        hit = False
        for g in genes_str.split(","):
            g = g.strip()
            if g in gene_idx:
                col[gene_idx[g]] = 1.0
                hit = True
        if hit:
            cols.append(col)
    return np.stack(cols, axis=1)   # (G, P)


class MaskedLinear(nn.Module):
    """Linear whose weight is masked to the allowed (gene, pathway) connections."""
    def __init__(self, mask):
        super().__init__()
        self.register_buffer("mask", torch.tensor(mask, dtype=torch.float32))
        self.weight = nn.Parameter(torch.randn(mask.shape) * 0.01)
        self.bias = nn.Parameter(torch.zeros(mask.shape[1]))

    def forward(self, x):
        return x @ (self.weight * self.mask) + self.bias


class PNetSurv(nn.Module):
    def __init__(self, mask, n_modalities, dropout=0.3):
        super().__init__()
        G, P = mask.shape
        # Per-gene layer: combine the n_modalities data types per gene.
        self.gene_w = nn.Parameter(torch.randn(G, n_modalities) * 0.1)
        self.gene_b = nn.Parameter(torch.zeros(G))
        self.gene_to_pw = MaskedLinear(mask)
        self.pw_layers = nn.Sequential(
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(P, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1, bias=False),   # Cox PH head
        )

    def forward(self, x):           # x: (B, G, n_modalities)
        gene = (x * self.gene_w).sum(-1) + self.gene_b   # (B, G)
        gene = torch.tanh(gene)
        pw = self.gene_to_pw(gene)                        # (B, P)
        return self.pw_layers(pw).squeeze(-1)


def iterate_minibatches(n, batch_size, device, shuffle=True):
    idx = torch.randperm(n, device=device) if shuffle else torch.arange(n, device=device)
    for b in range(0, n, batch_size):
        yield idx[b:b + batch_size]


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    setup_seed(SEED)

    print("Loading 2-modality survival data...")
    mods, time_np, event_np, patients, genes = D.load_modalities()
    print(f"patients={len(patients)} genes={len(genes)} events={int(event_np.sum())}")

    mask = build_mask(genes, pd.read_csv(PATHWAY_FILE))
    print(f"gene->pathway mask: {mask.shape}  density={mask.mean():.4f}")

    folds = D.make_folds(event_np)
    if SMOKE:
        folds = folds[:1]

    batch_size = D.BATCH_SIZE
    epochs = D.MAX_EPOCHS
    min_epochs = D.MIN_EPOCHS
    patience = D.PATIENCE
    lr, weight_decay = D.LR, D.WEIGHT_DECAY

    per_fold = []
    risk_records = []
    for fold_i, (tr_idx, val_idx, test_idx) in enumerate(folds, 1):
        print(f"\n{'='*60}\nFOLD {fold_i}/{len(folds)}\n{'='*60}")
        setup_seed(SEED + fold_i)

        # Continuous modalities z-scored on train; mutation binary.
        # Stack in MODALITY_NAMES order -> (N, G, n_modalities).
        layers = []
        for name in D.MODALITY_NAMES:
            a = mods[name].astype(np.float32)
            if name != "mutation":
                a = StandardScaler().fit(a[tr_idx]).transform(a).astype(np.float32)
            layers.append(a)
        X = np.stack(layers, axis=-1)
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        time_t = torch.tensor(time_np, dtype=torch.float32, device=device)
        event_t = torch.tensor(event_np, dtype=torch.float32, device=device)

        tr = torch.tensor(tr_idx, dtype=torch.long, device=device)
        va = torch.tensor(val_idx, dtype=torch.long, device=device)
        te = torch.tensor(test_idx, dtype=torch.long, device=device)

        model = PNetSurv(mask, n_modalities=X.shape[2]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_c, no_improve, best_state = -np.inf, 0, None
        for epoch in range(1, epochs + 1):
            model.train()
            for bidx in iterate_minibatches(len(tr), batch_size, device):
                sel = tr[bidx]
                opt.zero_grad(set_to_none=True)
                loss = cox_partial_loglik_loss(model(X_t[sel]), time_t[sel], event_t[sel])
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
            model.eval()
            with torch.no_grad():
                vc = harrell_c_index(model(X_t[va]).cpu().numpy(),
                                     time_np[val_idx], event_np[val_idx])
            if not np.isnan(vc) and vc > best_val_c:
                best_val_c, no_improve = vc, 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1
            if epoch >= min_epochs and no_improve >= patience:
                print(f"  early stop @ epoch {epoch} (best val C={best_val_c:.4f})")
                break

        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.eval()
        with torch.no_grad():
            risk = model(X_t[te]).cpu().numpy()
        m = fold_metrics(risk, time_np[test_idx], event_np[test_idx])
        per_fold.append(m)
        risk_records.append({"test_idx": test_idx, "risk": risk,
                             "time": time_np[test_idx], "event": event_np[test_idx]})
        print(f"  Fold {fold_i} C-index={m['c_index']:.4f} (n={m['n']}, events={m['events']})")
        del model, X_t
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_metrics_json(RESULTS_JSON, "pnet", os.environ.get("SURV_DATASET_TAG", "pan_survival_cox"), per_fold)
    write_risk_scores(RISK_CSV, patients, risk_records)
    print(f"\nWrote metrics to {RESULTS_JSON}")
    print(f"Wrote risk scores to {RISK_CSV}")


if __name__ == "__main__":
    run()
