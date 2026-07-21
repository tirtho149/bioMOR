"""Early Integration CNN for 2-modality pan-cancer Cox survival prediction.

Concatenates cnv + mutation along features -> (N, 2*G) and feeds the early-
integration CNN architecture, but the head emits a single log-relative-hazard
(DeepSurv-style) trained with the negative Cox partial log-likelihood.
Metric: Harrell's C-index. Uses the shared survival data contract and 5-fold
stratified splits.

Run from this directory (or with this dir on PYTHONPATH):
    python cnn_ei_survival.py
Set SMOKE=1 for a fast 1-fold / few-epoch smoke test.
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

RESULTS_JSON = os.path.join(HERE, os.environ.get("SURV_RESULTS_DIR", "results"), "CNN_ei_survival_metrics.json")
RISK_CSV = os.path.join(HERE, os.environ.get("SURV_RESULTS_DIR", "results"), "CNN_ei_risk_scores.csv")
SMOKE = os.environ.get("SMOKE", "0") == "1"


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class EarlyIntegrationCNN(nn.Module):
    """Early Integration CNN with a single-scalar Cox (risk) head."""
    def __init__(self, in_dim):
        super().__init__()
        conv1_out = in_dim - 1000 + 1
        pool1_out = conv1_out // 100
        conv2_out = pool1_out - 50 + 1
        pool2_out = conv2_out // 10
        linear_input = pool2_out * 16

        self.FC = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=1000),
            nn.ReLU(),
            nn.MaxPool1d(100),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=50),
            nn.ReLU(),
            nn.MaxPool1d(10),
            nn.Flatten(),
            nn.Linear(in_features=int(linear_input), out_features=50),
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=1, bias=False),  # Cox PH head
        )

    def forward(self, x):
        x = x.unsqueeze(1)            # add channel dim
        return self.FC(x).squeeze(-1)  # (B,) log-relative-hazard


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
    """Per-modality preprocessing then concat in MODALITY_NAMES order:
    mutation kept binary, every other (continuous) modality z-scored on train."""
    parts = []
    for name in D.MODALITY_NAMES:
        x = mods[name][idx]
        if name != "mutation":
            if fit:
                scalers[name].fit(x)
            x = scalers[name].transform(x)
        parts.append(x.astype(np.float32))
    return np.concatenate(parts, axis=1).astype(np.float32), scalers


def run():
    batch_size = D.BATCH_SIZE
    epochs = D.MAX_EPOCHS
    lr = D.LR
    weight_decay = D.WEIGHT_DECAY
    patience = D.PATIENCE
    min_epochs = D.MIN_EPOCHS

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    setup_seed(42)

    print("Loading 2-modality survival data...")
    mods, time, event, patients, genes = D.load_modalities()
    in_dim = len(D.MODALITY_NAMES) * len(genes)
    print(f"patients={len(patients)} genes={len(genes)} events={int(event.sum())} in_dim={in_dim}")

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

        train_loader = DataLoader(
            SurvDataset(X_train, time[train_idx], event[train_idx]),
            batch_size=batch_size, shuffle=True, drop_last=True,
            pin_memory=(device.type == 'cuda'))
        val_loader = DataLoader(SurvDataset(X_val, time[val_idx], event[val_idx]),
                                batch_size=batch_size)
        test_loader = DataLoader(SurvDataset(X_test, time[test_idx], event[test_idx]),
                                 batch_size=batch_size)

        model = EarlyIntegrationCNN(in_dim=in_dim).to(device)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_c, no_improve, best_state = -np.inf, 0, None
        for epoch in range(1, epochs + 1):
            model.train()
            for data, t_b, e_b in train_loader:
                data, t_b, e_b = data.to(device), t_b.to(device), e_b.to(device)
                optimizer.zero_grad()
                risk = model(data)
                loss = cox_partial_loglik_loss(risk, t_b, e_b)
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()

            val_c = evaluate(model, val_loader, device)
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
        risk, t_arr, e_arr = predict(model, test_loader, device)
        m = fold_metrics(risk, t_arr, e_arr)
        per_fold.append(m)
        risk_records.append({"test_idx": test_idx, "risk": risk,
                             "time": t_arr, "event": e_arr})
        print(f"  Fold {fold + 1} C-index={m['c_index']:.4f} (n={m['n']}, events={m['events']})")
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    write_metrics_json(RESULTS_JSON, "CNN_ei", os.environ.get("SURV_DATASET_TAG", "pan_survival_cox"), per_fold)
    write_risk_scores(RISK_CSV, patients, risk_records)
    print(f"\nWrote metrics to {RESULTS_JSON}")
    print(f"Wrote risk scores to {RISK_CSV}")


@torch.no_grad()
def evaluate(model, loader, device):
    risk, t_arr, e_arr = predict(model, loader, device)
    return harrell_c_index(risk, t_arr, e_arr)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    risks, times, events = [], [], []
    for data, t_b, e_b in loader:
        r = model(data.to(device))
        risks.append(r.cpu().numpy())
        times.append(t_b.numpy())
        events.append(e_b.numpy())
    return (np.concatenate(risks), np.concatenate(times), np.concatenate(events))


if __name__ == "__main__":
    run()
