"""MOGAT for the 3-modality, 5-class TCGA BRCA PAM50 task.

Self-contained revision of the MOGAT pipeline (data_preprocessor.py +
mogat_adapted.py) for 3 modalities (cnv + expression + mutation) and 5-class
PAM50 classification. Keeps MOGAT's design: one GAT (lib.module2.Net) learns a
node embedding per modality from a patient-similarity graph, the three
embeddings are concatenated with the raw features, and an MLP integrates them.

Changes vs. the original binary/2-modality code:
  * node_networks = ['cna', 'mut', 'expr'] (was ['cna', 'mut'])
  * MLP head num_classes = 5 with full-softmax probabilities (was [:, 1])
  * multiclass metrics via multiclass_metrics.fold_metrics
  * uses the shared data contract (same patients + 5-fold splits as the others)

The GAT graph is transductive (built over all patients, like the original);
train/val/test masks come from the shared 5-fold splits.

Run from baseline_MOGAT/:
    python mogat_3mod_multiclass.py
Set SMOKE=1 for a fast 1-fold / few-epoch smoke test.
"""
import os
import sys
import gc
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import brca_pam50_data as D
from multiclass_metrics import fold_metrics, write_metrics_json
from lib import module2

RESULTS_JSON = os.path.join(HERE, "results", "brca_pam50", "MOGAT_metrics.json")
SMOKE = os.environ.get("SMOKE", "0") == "1"

NODE_NETWORKS = ["cna", "mut", "expr"]      # cnv, mutation, expression
MODALITY_OF = {"cna": "cnv", "mut": "mutation", "expr": "expression"}
TOP_K = 5
RANDOM_STATE = D.SEED

# GAT embedding hyperparameters (mirror mogat_adapted.py).
MAX_EPOCHS = 4 if SMOKE else 200
MIN_EPOCHS = 1 if SMOKE else 50
PATIENCE = 25
GAT_LR = 0.0001
GAT_HID = 512

# MLP integration head.
HP_TRIALS = 2 if SMOKE else 20
MLP_MAX_EPOCHS = 20 if SMOKE else 500
MLP_PATIENCE = 50
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


# ---------------------------------------------------------------------------
# Patient-similarity graphs (transductive, built over all patients).
# ---------------------------------------------------------------------------
def edges_from_similarity(similarity, top_k=TOP_K):
    """Top-k positive-similarity neighbours per patient -> (edge_index, weights)."""
    n = similarity.shape[0]
    src, dst, w = [], [], []
    for i in range(n):
        sims = similarity[i].copy()
        sims[i] = -np.inf
        top = np.argsort(sims)[-top_k:]
        for j in top:
            if sims[j] > 0:
                src.append(i)
                dst.append(j)
                w.append(sims[j])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(w, dtype=torch.float32)
    return edge_index, edge_attr


def jaccard_matrix(binary):
    n = binary.shape[0]
    b = (binary != 0)
    out = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        inter = (b[i] & b).sum(axis=1)
        union = (b[i] | b).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.where(union > 0, inter / union, 0.0)
        out[i] = sim
    return out


def build_graphs(feats_norm):
    """feats_norm: dict net -> (N, G) normalized features. Returns dict net -> (edge_index, edge_attr)."""
    graphs = {}
    for net, X in feats_norm.items():
        if net == "mut":
            sim = jaccard_matrix(X)
        else:
            sim = np.corrcoef(X)
            sim = np.nan_to_num(sim, nan=0.0)
        graphs[net] = edges_from_similarity(sim)
        print(f"  {net} graph: {graphs[net][0].shape[1]} edges")
    return graphs


# ---------------------------------------------------------------------------
# GAT embedding training (transductive).
# ---------------------------------------------------------------------------
def train_gat_embedding(node_x, edge_index, edge_attr, y, train_mask, valid_mask,
                        out_size, device):
    setup_seed(RANDOM_STATE)
    data = Data(x=node_x, edge_index=edge_index, edge_attr=edge_attr, y=y).to(device)
    data.train_mask = train_mask
    data.valid_mask = valid_mask

    model = module2.Net(in_size=node_x.shape[1], hid_size=GAT_HID, out_size=out_size).to(device)
    optimizer = Adam(model.parameters(), lr=GAT_LR)
    criterion = nn.CrossEntropyLoss()

    best_valid = np.inf
    patience_count = 0
    selected_emb = None
    for epoch in range(MAX_EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out, emb = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out, emb = model(data)
            v_loss = criterion(out[data.valid_mask], data.y[data.valid_mask]).item()
        if v_loss < best_valid:
            best_valid = v_loss
            patience_count = 0
            selected_emb = emb.detach().clone()
        else:
            patience_count += 1
        if epoch >= MIN_EPOCHS and patience_count >= PATIENCE:
            break

    del model, optimizer, data
    clear_gpu_memory()
    return selected_emb


# ---------------------------------------------------------------------------
# MLP integration head (multiclass).
# ---------------------------------------------------------------------------
class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout=0.2):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def train_mlp(model, X_tr, y_tr, X_val, y_val, lr, device,
              max_epochs=MLP_MAX_EPOCHS, patience=MLP_PATIENCE):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    batch_size = min(16, X_tr.shape[0])
    n_batches = (X_tr.shape[0] + batch_size - 1) // batch_size

    best_val = np.inf
    patience_count = 0
    best_state = None
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(X_tr.shape[0], device=device)
        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(X_tr[idx]), y_tr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            v_loss = criterion(model(X_val), y_val).item()
        if v_loss < best_val:
            best_val = v_loss
            patience_count = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
        if patience_count >= patience:
            break
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model


def hp_search_mlp(X_tr, y_tr, X_val, y_val, num_classes, device, n_trials=HP_TRIALS):
    setup_seed(RANDOM_STATE)
    hidden_configs = [[16], [32], [64], [128], [256], [512],
                      [32, 32], [64, 32], [128, 64], [256, 128]]
    lrs = [0.1, 0.01, 0.001, 0.0001]
    dropouts = [0.3, 0.5, 0.7]
    y_val_np = y_val.cpu().numpy()

    best_f1 = -1.0
    best = {"hidden": [128], "lr": 0.001, "dropout": 0.5}
    for _ in range(n_trials):
        hidden = hidden_configs[np.random.randint(len(hidden_configs))]
        lr = lrs[np.random.randint(len(lrs))]
        dropout = dropouts[np.random.randint(len(dropouts))]
        model = MLPClassifier(X_tr.shape[1], hidden, num_classes, dropout).to(device)
        model = train_mlp(model, X_tr, y_tr, X_val, y_val, lr, device,
                          max_epochs=100, patience=15)
        model.eval()
        with torch.no_grad():
            preds = model(X_val).argmax(dim=1).cpu().numpy()
        f1 = f1_score(y_val_np, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best = {"hidden": hidden, "lr": lr, "dropout": dropout}
        del model
        clear_gpu_memory()
    print(f"  best val macro-F1={best_f1:.4f} params={best}")
    return best


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    setup_seed(RANDOM_STATE)

    num_classes = D.N_CLASSES
    print("Loading 3-modality data...")
    mods, y, patients, genes = D.load_modalities()
    print(f"Patients={len(patients)} genes={len(genes)} classes={num_classes}")

    folds = D.make_folds(y)
    if SMOKE:
        folds = folds[:1]

    per_fold = []
    for fold, (train_idx, val_idx, test_idx) in enumerate(folds):
        print(f"\n{'='*60}\nFold {fold + 1}/{len(folds)}\n{'='*60}")
        clear_gpu_memory()

        # Per-fold standardization fit on train rows (cna, expr only; mut binary).
        feats_norm = {}
        for net in NODE_NETWORKS:
            X = mods[MODALITY_OF[net]].astype(np.float32)
            if net == "mut":
                feats_norm[net] = X
            else:
                scaler = StandardScaler()
                Xn = X.copy()
                scaler.fit(X[train_idx])
                Xn = scaler.transform(X).astype(np.float32)
                feats_norm[net] = Xn

        print("Building patient-similarity graphs...")
        graphs = build_graphs(feats_norm)

        # Shared node features: all modalities concatenated (as in original MOGAT).
        node_x = torch.tensor(
            np.concatenate([feats_norm[n] for n in NODE_NETWORKS], axis=1),
            dtype=torch.float32).to(device)
        y_gpu = torch.tensor(y, dtype=torch.long).to(device)

        n = len(y)
        train_mask = torch.zeros(n, dtype=torch.bool, device=device)
        valid_mask = torch.zeros(n, dtype=torch.bool, device=device)
        train_mask[torch.tensor(train_idx, dtype=torch.long)] = True
        valid_mask[torch.tensor(val_idx, dtype=torch.long)] = True

        # Train a GAT embedding per modality graph.
        embeddings = []
        for net in NODE_NETWORKS:
            print(f"Training GAT embedding for {net}...")
            edge_index, edge_attr = graphs[net]
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device)
            emb = train_gat_embedding(node_x, edge_index, edge_attr, y_gpu,
                                      train_mask, valid_mask, num_classes, device)
            embeddings.append(emb)

        # Integration: concat embeddings (+ raw features) -> MLP.
        integrated = torch.cat(embeddings, dim=1)
        if ADD_RAW_FEAT:
            integrated = torch.cat([integrated, node_x], dim=1)
        print(f"Integrated feature dim: {integrated.shape[1]}")

        tr = torch.tensor(train_idx, dtype=torch.long, device=device)
        va = torch.tensor(val_idx, dtype=torch.long, device=device)
        te = torch.tensor(test_idx, dtype=torch.long, device=device)
        X_tr, X_val, X_test = integrated[tr], integrated[va], integrated[te]
        y_tr, y_val, y_test = y_gpu[tr], y_gpu[va], y_gpu[te]

        best = hp_search_mlp(X_tr, y_tr, X_val, y_val, num_classes, device)
        model = MLPClassifier(X_tr.shape[1], best["hidden"], num_classes,
                              best["dropout"]).to(device)
        model = train_mlp(model, X_tr, y_tr, X_val, y_val, best["lr"], device)

        model.eval()
        with torch.no_grad():
            probs = F.softmax(model(X_test), dim=1).cpu().numpy()
        t_true = y_test.cpu().numpy()

        m = fold_metrics(t_true, probs, n_classes=num_classes)
        per_fold.append(m)
        print(f"  Fold {fold + 1} acc={m['accuracy']:.4f} f1={m['f1']:.4f} "
              f"f1_macro={m['f1_macro']:.4f} auc={m['auc']:.4f}")

        del model, integrated, node_x
        clear_gpu_memory()

    write_metrics_json(RESULTS_JSON, "MOGAT", "brca_pam50", num_classes, per_fold)
    print(f"\nWrote metrics to {RESULTS_JSON}")


if __name__ == "__main__":
    run()
