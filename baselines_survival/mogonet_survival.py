"""MOGONET for 2-modality pan-cancer Cox survival prediction.

Survival revision of MOGONET. Reuses MOGONET's original per-view GCN encoder
(baseline_MOGONET/models.py: GCN_E) on a cosine-kNN patient-similarity graph
per modality; the per-view node embeddings are concatenated and fed to an MLP
that emits a single log-relative-hazard, trained with the Cox partial
log-likelihood. (The original VCDN/Classifier machinery is class-count-coupled
-- num_class^num_view -- so it is replaced by the Cox head for survival, while
the GCN-on-patient-graph core is kept.) Metric: Harrell's C-index.

    python mogonet_survival.py     (SMOKE=1 for a fast 1-fold smoke test)
"""
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "baseline_MOGONET"))  # for models.py

import pan_survival_data as D
from survival_metrics import cox_partial_loglik_loss, harrell_c_index, fold_metrics, write_metrics_json, write_risk_scores
from models import GCN_E

RESULTS_JSON = os.path.join(HERE, os.environ.get("SURV_RESULTS_DIR", "results"), "MOGONET_survival_metrics.json")
RISK_CSV = os.path.join(HERE, os.environ.get("SURV_RESULTS_DIR", "results"), "MOGONET_risk_scores.csv")
SMOKE = os.environ.get("SMOKE", "0") == "1"

MODALITY_NAMES = D.MODALITY_NAMES  # 2- or 3-modal depending on SURV_3MODAL
ADJ_K = 10
DIM_HE = [400, 400, 200]
GCN_DROPOUT = 0.2
LR = D.LR                 # shared learning rate
WEIGHT_DECAY = D.WEIGHT_DECAY
# Transductive full-graph GCN: keeps its native (longer) epoch schedule by
# design, but shares the lr / weight-decay / val-C-index early-stopping rule.
MAX_EPOCHS = 10 if SMOKE else 2500
EVAL_INTERVAL = 5 if SMOKE else 50
PATIENCE = 25          # in eval-intervals
MIN_EPOCHS = 5 if SMOKE else 200
SEED = D.SEED


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def build_adj(X, k=ADJ_K, device="cpu"):
    """Symmetric-normalized cosine-kNN adjacency (dense [N,N])."""
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    Xn = Xt / (Xt.norm(dim=1, keepdim=True) + 1e-8)
    S = Xn @ Xn.T                              # cosine similarity
    S.fill_diagonal_(-1.0)
    topv, topi = S.topk(k, dim=1)              # k nearest neighbours
    A = torch.zeros_like(S)
    A.scatter_(1, topi, torch.clamp(topv, min=0.0))
    A = torch.maximum(A, A.T)                  # symmetrize
    A.fill_diagonal_(1.0)                      # self loops
    d = A.sum(1)
    dinv = torch.pow(d.clamp(min=1e-8), -0.5)
    return (dinv.unsqueeze(1) * A) * dinv.unsqueeze(0)


class MogonetSurv(nn.Module):
    def __init__(self, dim_list, dim_he=DIM_HE, dropout=GCN_DROPOUT):
        super().__init__()
        self.encoders = nn.ModuleList([GCN_E(d, dim_he, dropout) for d in dim_list])
        fused = dim_he[-1] * len(dim_list)
        self.head = nn.Sequential(
            nn.Linear(fused, 128), nn.LeakyReLU(0.25), nn.Dropout(dropout),
            nn.Linear(128, 1, bias=False),
        )

    def forward(self, x_list, adj_list):
        embs = [enc(x, adj) for enc, x, adj in zip(self.encoders, x_list, adj_list)]
        return self.head(torch.cat(embs, dim=1)).squeeze(-1)


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    setup_seed(SEED)

    print("Loading 2-modality survival data...")
    mods, time_np, event_np, patients, genes = D.load_modalities()
    print(f"patients={len(patients)} genes={len(genes)} events={int(event_np.sum())}")

    folds = D.make_folds(event_np)
    if SMOKE:
        folds = folds[:1]

    time_all = torch.tensor(time_np, dtype=torch.float32, device=device)
    event_all = torch.tensor(event_np, dtype=torch.float32, device=device)

    per_fold = []
    risk_records = []
    for fold, (train_idx, val_idx, test_idx) in enumerate(folds):
        print(f"\n{'='*60}\nFold {fold + 1}/{len(folds)}\n{'='*60}")
        setup_seed(SEED + fold)

        # Per-view features: continuous modalities z-scored on train, mutation binary.
        x_list, adj_list, dim_list = [], [], []
        for name in MODALITY_NAMES:
            X = mods[name].astype(np.float32)
            if name != "mutation":
                X = StandardScaler().fit(X[train_idx]).transform(X).astype(np.float32)
            x_list.append(torch.tensor(X, dtype=torch.float32, device=device))
            adj_list.append(build_adj(X, device=device))
            dim_list.append(X.shape[1])

        tr = torch.tensor(train_idx, dtype=torch.long, device=device)
        va = torch.tensor(val_idx, dtype=torch.long, device=device)

        model = MogonetSurv(dim_list).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        best_val_c, no_improve, best_state = -np.inf, 0, None
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            opt.zero_grad(set_to_none=True)
            risk = model(x_list, adj_list)
            loss = cox_partial_loglik_loss(risk[tr], time_all[tr], event_all[tr])
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()

            if epoch % EVAL_INTERVAL == 0:
                model.eval()
                with torch.no_grad():
                    risk = model(x_list, adj_list)
                    vc = harrell_c_index(risk[va].cpu().numpy(),
                                         time_np[val_idx], event_np[val_idx])
                if not np.isnan(vc) and vc > best_val_c:
                    best_val_c, no_improve = vc, 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    no_improve += 1
                if epoch >= MIN_EPOCHS and no_improve >= PATIENCE:
                    print(f"  early stop @ epoch {epoch} (best val C={best_val_c:.4f})")
                    break

        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.eval()
        with torch.no_grad():
            risk = model(x_list, adj_list).cpu().numpy()
        m = fold_metrics(risk[test_idx], time_np[test_idx], event_np[test_idx])
        per_fold.append(m)
        risk_records.append({"test_idx": test_idx, "risk": risk[test_idx],
                             "time": time_np[test_idx], "event": event_np[test_idx]})
        print(f"  Fold {fold + 1} C-index={m['c_index']:.4f} (n={m['n']}, events={m['events']})")
        del model, x_list, adj_list
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_metrics_json(RESULTS_JSON, "MOGONET", os.environ.get("SURV_DATASET_TAG", "pan_survival_cox"), per_fold)
    write_risk_scores(RISK_CSV, patients, risk_records)
    print(f"\nWrote metrics to {RESULTS_JSON}")
    print(f"Wrote risk scores to {RISK_CSV}")


if __name__ == "__main__":
    run()
