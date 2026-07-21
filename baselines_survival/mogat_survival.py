"""MOGAT for 2-modality pan-cancer Cox survival prediction.

Self-contained survival revision of MOGAT. Keeps MOGAT's design: one GAT
(lib.module2.Net) learns a node embedding per modality from a patient-similarity
graph, the embeddings are concatenated with the raw features, and an MLP
integrates them. Both the GAT (out_size=1) and the MLP emit a single
log-relative-hazard, trained with the Cox partial log-likelihood. Metric:
Harrell's C-index.

Changes vs. the 3-modality PAM50 revision:
  * node_networks = ['cna', 'mut'] (no expression in the survival cohort)
  * patient-similarity built with matmul (Pearson for cnv, Jaccard for mut)
    so it scales to ~8.8k patients (the original O(n^2 * G) loop does not)
  * single-risk Cox head + C-index instead of softmax + multiclass metrics

    python mogat_survival.py     (SMOKE=1 for a fast 1-fold smoke test)
"""
import os
import sys
import gc
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "baseline_MOGAT"))  # for lib.module2

import pan_survival_data as D
from survival_metrics import cox_partial_loglik_loss, harrell_c_index, fold_metrics, write_metrics_json, write_risk_scores
from lib import module2

RESULTS_JSON = os.path.join(HERE, os.environ.get("SURV_RESULTS_DIR", "results"), "MOGAT_survival_metrics.json")
RISK_CSV = os.path.join(HERE, os.environ.get("SURV_RESULTS_DIR", "results"), "MOGAT_risk_scores.csv")
SMOKE = os.environ.get("SMOKE", "0") == "1"

# Map data-loader modality name -> MOGAT node-network name. Derived from the
# active modality set so the 3-modal (SURV_3MODAL=1) run adds an expression view.
_NET_OF = {"cnv": "cna", "mutation": "mut", "expression": "exp"}
NODE_NETWORKS = [_NET_OF[m] for m in D.MODALITY_NAMES]
MODALITY_OF = {_NET_OF[m]: m for m in D.MODALITY_NAMES}
TOP_K = 5
RANDOM_STATE = D.SEED

# Two-stage (GAT embedding -> risk head) but every tunable knob is the shared
# one: no per-fold hyperparameter search (removed) so it's apples-to-apples.
MAX_EPOCHS = D.MAX_EPOCHS
MIN_EPOCHS = D.MIN_EPOCHS
PATIENCE = D.PATIENCE
GAT_LR = D.LR
GAT_HID = 256

# Fixed risk-head architecture (previously the default of the removed search).
MLP_HIDDEN = [128]
MLP_DROPOUT = 0.3
MLP_MAX_EPOCHS = D.MAX_EPOCHS
MLP_MIN_EPOCHS = D.MIN_EPOCHS
MLP_PATIENCE = D.PATIENCE
ADD_RAW_FEAT = True


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# --- patient-similarity (matmul-based, scales to ~8.8k patients) ------------
def edges_from_similarity(similarity, top_k=TOP_K):
    n = similarity.shape[0]
    src, dst, w = [], [], []
    for i in range(n):
        sims = similarity[i].copy()
        sims[i] = -np.inf
        top = np.argpartition(sims, -top_k)[-top_k:]
        for j in top:
            if sims[j] > 0:
                src.append(i); dst.append(int(j)); w.append(float(sims[j]))
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(w, dtype=torch.float32)
    return edge_index, edge_attr


def pearson_matrix(X):
    """Row-wise Pearson correlation via matmul (X: N x G)."""
    Xc = X - X.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(Xc, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    Xn = (Xc / norm).astype(np.float32)
    return np.clip(Xn @ Xn.T, -1.0, 1.0)


def jaccard_matrix(binary):
    """Jaccard similarity via matmul (binary: N x G)."""
    B = (binary != 0).astype(np.float32)
    inter = B @ B.T
    rs = B.sum(axis=1)
    union = rs[:, None] + rs[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(union > 0, inter / union, 0.0)
    return sim.astype(np.float32)


def build_graphs(feats_norm):
    graphs = {}
    for net, X in feats_norm.items():
        sim = jaccard_matrix(X) if net == "mut" else pearson_matrix(X)
        sim = np.nan_to_num(sim, nan=0.0)
        graphs[net] = edges_from_similarity(sim)
        print(f"  {net} graph: {graphs[net][0].shape[1]} edges")
        del sim
    return graphs


# --- GAT embedding (Cox-trained, out_size=1) --------------------------------
def train_gat_embedding(node_x, edge_index, edge_attr, time, event,
                        train_mask, valid_mask, device):
    setup_seed(RANDOM_STATE)
    data = Data(x=node_x, edge_index=edge_index, edge_attr=edge_attr).to(device)
    model = module2.Net(in_size=node_x.shape[1], hid_size=GAT_HID, out_size=1).to(device)
    optimizer = Adam(model.parameters(), lr=GAT_LR)

    best_valid, patience_count, selected_emb = -np.inf, 0, None
    for epoch in range(MAX_EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out, emb = model(data)
        risk = out.squeeze(-1)
        loss = cox_partial_loglik_loss(risk[train_mask], time[train_mask], event[train_mask])
        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            out, emb = model(data)
            vc = harrell_c_index(out.squeeze(-1)[valid_mask].cpu().numpy(),
                                 time[valid_mask].cpu().numpy(),
                                 event[valid_mask].cpu().numpy())
        if not np.isnan(vc) and vc > best_valid:
            best_valid, patience_count = vc, 0
            selected_emb = emb.detach().clone()
        else:
            patience_count += 1
        if epoch >= MIN_EPOCHS and patience_count >= PATIENCE:
            break

    if selected_emb is None:
        selected_emb = emb.detach().clone()
    del model, optimizer, data
    clear_gpu_memory()
    return selected_emb


# --- MLP integration head (Cox) ---------------------------------------------
class MLPRisk(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout=0.3):
        super().__init__()
        layers, prev = [], input_size
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1, bias=False))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze(-1)


def train_mlp(model, X_tr, t_tr, e_tr, X_val, t_val, e_val, lr, device,
              max_epochs=MLP_MAX_EPOCHS, patience=MLP_PATIENCE,
              min_epochs=MLP_MIN_EPOCHS):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=D.WEIGHT_DECAY)
    batch_size = min(D.BATCH_SIZE, X_tr.shape[0])
    n_batches = (X_tr.shape[0] + batch_size - 1) // batch_size
    best_val, patience_count, best_state = -np.inf, 0, None
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(X_tr.shape[0], device=device)
        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = cox_partial_loglik_loss(model(X_tr[idx]), t_tr[idx], e_tr[idx])
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            vc = harrell_c_index(model(X_val).cpu().numpy(),
                                 t_val.cpu().numpy(), e_val.cpu().numpy())
        if not np.isnan(vc) and vc > best_val:
            best_val, patience_count = vc, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
        if epoch + 1 >= min_epochs and patience_count >= patience:
            break
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model, best_val


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    setup_seed(RANDOM_STATE)

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
        clear_gpu_memory()

        feats_norm = {}
        for net in NODE_NETWORKS:
            X = mods[MODALITY_OF[net]].astype(np.float32)
            if net == "mut":
                feats_norm[net] = X
            else:
                scaler = StandardScaler().fit(X[train_idx])
                feats_norm[net] = scaler.transform(X).astype(np.float32)

        print("Building patient-similarity graphs...")
        graphs = build_graphs(feats_norm)

        node_x = torch.tensor(
            np.concatenate([feats_norm[n] for n in NODE_NETWORKS], axis=1),
            dtype=torch.float32).to(device)

        n = len(time_np)
        train_mask = torch.zeros(n, dtype=torch.bool, device=device)
        valid_mask = torch.zeros(n, dtype=torch.bool, device=device)
        train_mask[torch.tensor(train_idx, dtype=torch.long)] = True
        valid_mask[torch.tensor(val_idx, dtype=torch.long)] = True

        embeddings = []
        for net in NODE_NETWORKS:
            print(f"Training GAT embedding for {net}...")
            edge_index, edge_attr = graphs[net]
            emb = train_gat_embedding(node_x, edge_index.to(device), edge_attr.to(device),
                                      time_all, event_all, train_mask, valid_mask, device)
            embeddings.append(emb)

        integrated = torch.cat(embeddings, dim=1)
        if ADD_RAW_FEAT:
            integrated = torch.cat([integrated, node_x], dim=1)
        print(f"Integrated feature dim: {integrated.shape[1]}")

        tr = torch.tensor(train_idx, dtype=torch.long, device=device)
        va = torch.tensor(val_idx, dtype=torch.long, device=device)
        te = torch.tensor(test_idx, dtype=torch.long, device=device)
        X_tr, X_val, X_test = integrated[tr], integrated[va], integrated[te]
        t_tr, t_val = time_all[tr], time_all[va]
        e_tr, e_val = event_all[tr], event_all[va]

        # Fixed head + shared lr (no per-fold hyperparameter search) for a fair
        # apples-to-apples comparison with the other baselines.
        setup_seed(RANDOM_STATE)
        model = MLPRisk(X_tr.shape[1], MLP_HIDDEN, MLP_DROPOUT).to(device)
        model, _ = train_mlp(model, X_tr, t_tr, e_tr, X_val, t_val, e_val, D.LR, device)

        model.eval()
        with torch.no_grad():
            risk = model(X_test).cpu().numpy()
        m = fold_metrics(risk, time_np[test_idx], event_np[test_idx])
        per_fold.append(m)
        risk_records.append({"test_idx": test_idx, "risk": risk,
                             "time": time_np[test_idx], "event": event_np[test_idx]})
        print(f"  Fold {fold + 1} C-index={m['c_index']:.4f} (n={m['n']}, events={m['events']})")

        del model, integrated, node_x
        clear_gpu_memory()

    write_metrics_json(RESULTS_JSON, "MOGAT", os.environ.get("SURV_DATASET_TAG", "pan_survival_cox"), per_fold)
    write_risk_scores(RISK_CSV, patients, risk_records)
    print(f"\nWrote metrics to {RESULTS_JSON}")
    print(f"Wrote risk scores to {RISK_CSV}")


if __name__ == "__main__":
    run()
