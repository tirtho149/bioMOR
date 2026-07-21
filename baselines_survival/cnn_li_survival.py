"""Late Integration CNN for 2-modality pan-cancer Cox survival prediction.

One conv branch per modality (cnv, mutation), merged by an MLP that emits a
single log-relative-hazard (DeepSurv-style), trained with the negative Cox
partial log-likelihood. Metric: Harrell's C-index. Shared survival contract
and 5-fold stratified splits.

    python cnn_li_survival.py     (SMOKE=1 for a fast 1-fold smoke test)
"""
import os
import random
import sys

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import warnings
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pan_survival_data as D
from survival_metrics import cox_partial_loglik_loss, harrell_c_index, fold_metrics, write_metrics_json, write_risk_scores

RESULTS_JSON = os.path.join(HERE, os.environ.get("SURV_RESULTS_DIR", "results"), "CNN_li_survival_metrics.json")
RISK_CSV = os.path.join(HERE, os.environ.get("SURV_RESULTS_DIR", "results"), "CNN_li_risk_scores.csv")
SMOKE = os.environ.get("SMOKE", "0") == "1"


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class LateIntegrationCNN(nn.Module):
    """One conv branch per modality, then a merge MLP -> single risk scalar."""
    def __init__(self, feature_dims):
        super().__init__()
        self.feature_dims = list(feature_dims)

        def branch():
            return nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=32, kernel_size=300),
                nn.ReLU(),
                nn.MaxPool1d(100),
                nn.Flatten(),
            )

        self.branches = nn.ModuleList([branch() for _ in self.feature_dims])
        total = sum(((d - 300 + 1) // 100) * 32 for d in self.feature_dims)
        self.FC_merge = nn.Sequential(
            nn.Linear(total, 100), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(100, 50), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(50, 10), nn.ReLU(),
            nn.Linear(10, 1, bias=False),  # Cox PH head
        )

    def forward(self, x):
        outs, start = [], 0
        for d, br in zip(self.feature_dims, self.branches):
            block = x[:, start:start + d].unsqueeze(1)
            outs.append(br(block))
            start += d
        return self.FC_merge(torch.cat(outs, dim=1)).squeeze(-1)


class SurvDataset(Dataset):
    def __init__(self, data, time, event):
        self.data, self.time, self.event = data, time, event

    def __getitem__(self, i):
        return (torch.from_numpy(self.data[i]).float(),
                torch.tensor(self.time[i]).float(),
                torch.tensor(self.event[i]).float())

    def __len__(self):
        return self.data.shape[0]


def integrate(mods, idx, scalers=None, fit=False):
    """Concat modalities in MODALITY_NAMES order (matches feature_dims): mutation
    kept binary, every other (continuous) modality z-scored on train."""
    parts = []
    for name in D.MODALITY_NAMES:
        x = mods[name][idx]
        if name != "mutation":
            if fit:
                scalers[name].fit(x)
            x = scalers[name].transform(x)
        parts.append(x.astype(np.float32))
    return np.concatenate(parts, axis=1).astype(np.float32), scalers


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    risks, times, events = [], [], []
    for data, t_b, e_b in loader:
        r = model(data.to(device))
        risks.append(r.cpu().numpy()); times.append(t_b.numpy()); events.append(e_b.numpy())
    return np.concatenate(risks), np.concatenate(times), np.concatenate(events)


def run():
    batch_size = D.BATCH_SIZE
    epochs = D.MAX_EPOCHS
    lr, weight_decay, patience = D.LR, D.WEIGHT_DECAY, D.PATIENCE
    min_epochs = D.MIN_EPOCHS

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    setup_seed(42)

    print("Loading 2-modality survival data...")
    mods, time, event, patients, genes = D.load_modalities()
    G = len(genes)
    feature_dims = [G] * len(D.MODALITY_NAMES)  # one branch per modality
    print(f"patients={len(patients)} genes={G} events={int(event.sum())}")

    folds = D.make_folds(event)
    if SMOKE:
        folds = folds[:1]

    per_fold = []
    risk_records = []
    for fold, (train_idx, val_idx, test_idx) in enumerate(folds):
        print(f"\nFold {fold + 1}/{len(folds)}")
        scalers = {n: StandardScaler() for n in D.MODALITY_NAMES if n != "mutation"}
        X_train, scalers = integrate(mods, train_idx, scalers, fit=True)
        X_val, _ = integrate(mods, val_idx, scalers)
        X_test, _ = integrate(mods, test_idx, scalers)

        train_loader = DataLoader(SurvDataset(X_train, time[train_idx], event[train_idx]),
                                  batch_size=batch_size, shuffle=True, drop_last=True,
                                  pin_memory=(device.type == 'cuda'))
        val_loader = DataLoader(SurvDataset(X_val, time[val_idx], event[val_idx]), batch_size=batch_size)
        test_loader = DataLoader(SurvDataset(X_test, time[test_idx], event[test_idx]), batch_size=batch_size)

        model = LateIntegrationCNN(feature_dims).to(device)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_c, no_improve, best_state = -np.inf, 0, None
        for epoch in range(1, epochs + 1):
            model.train()
            for data, t_b, e_b in train_loader:
                data, t_b, e_b = data.to(device), t_b.to(device), e_b.to(device)
                optimizer.zero_grad()
                loss = cox_partial_loglik_loss(model(data), t_b, e_b)
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
            val_c = harrell_c_index(*predict(model, val_loader, device))
            if not np.isnan(val_c) and val_c > best_val_c:
                best_val_c, no_improve = val_c, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1
            if epoch >= min_epochs and no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        te_risk, te_time, te_event = predict(model, test_loader, device)
        m = fold_metrics(te_risk, te_time, te_event)
        per_fold.append(m)
        risk_records.append({"test_idx": test_idx, "risk": te_risk,
                             "time": te_time, "event": te_event})
        print(f"  Fold {fold + 1} C-index={m['c_index']:.4f} (n={m['n']}, events={m['events']})")
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    write_metrics_json(RESULTS_JSON, "CNN_li", os.environ.get("SURV_DATASET_TAG", "pan_survival_cox"), per_fold)
    write_risk_scores(RISK_CSV, patients, risk_records)
    print(f"\nWrote metrics to {RESULTS_JSON}")
    print(f"Wrote risk scores to {RISK_CSV}")


if __name__ == "__main__":
    run()
