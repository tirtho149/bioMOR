"""Early Integration CNN for the 3-modality, 5-class TCGA BRCA PAM50 task.

Concatenates cnv + expression + mutation along features -> (N, 3*40543) and
feeds the existing early-integration CNN architecture (mimics eiCNN.py), with
num_classes=5 (softmax + CrossEntropyLoss). Uses the shared data contract and
the exact 5-fold splits from brca_pam50_data, and the multiclass metrics.

Run from the repo root:
    python baseline_CNN/eiCNN_3mod_multiclass.py
Set SMOKE=1 for a fast 1-fold / 2-epoch smoke test.
"""
import os
import random
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import warnings
warnings.filterwarnings('ignore')

import brca_pam50_data as D
from multiclass_metrics import fold_metrics, write_metrics_json

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_JSON = os.path.join(REPO, "baseline_CNN", "results", "brca_pam50", "CNN_ei_metrics.json")
SMOKE = os.environ.get("SMOKE", "0") == "1"


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class EarlyIntegrationCNN(nn.Module):
    """Early Integration CNN - same architecture as eiCNN baseline."""
    def __init__(self, in_dim, num_classes):
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
            nn.Linear(in_features=50, out_features=num_classes),
        )
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, num_classes):
        x = x.unsqueeze(1)  # add channel dim
        x = self.FC(x)
        if num_classes == 2:
            dec_logits = self.sigmoid(x)
        else:
            dec_logits = self.softmax(x)
        return dec_logits


class MultiOmicsDataset(Dataset):
    def __init__(self, data, label):
        super().__init__()
        self.data = data
        self.label = label

    def __getitem__(self, index):
        full_seq = torch.from_numpy(self.data[index]).float()
        return full_seq, self.label[index]

    def __len__(self):
        return self.data.shape[0]


class EarlyStopping:
    """Early stopping on a monitored validation score (higher is better)."""
    def __init__(self, patience=25, verbose=False, delta=0, stop=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        self.save_epoch = True
        self.delta = delta
        self.stop = stop

    def __call__(self, monitor, epoch):
        score = monitor[0] if len(monitor) == 1 else np.mean(monitor)
        if self.best_epoch is None:
            self.best_epoch = epoch
        if epoch <= self.stop:
            self.best_score = score
            self.early_stop = False
            self.best_epoch = epoch
            self.counter = 0
        if (self.best_score is None) | (epoch == 1):
            self.best_score = score
        elif score < self.best_score - self.delta:
            self.counter += 1
            self.save_epoch = False
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.save_epoch = True
            self.best_epoch = epoch


def macro_f1(y_true, y_prob):
    from sklearn.metrics import f1_score
    return f1_score(np.asarray(y_true), np.asarray(y_prob).argmax(axis=1),
                    average='macro', zero_division=0)


def run():
    # Hyperparameters mirror eiCNN.py
    batch_size = 16
    epochs = 2 if SMOKE else 200
    learning_rate = 0.0001
    weight_decay = 5e-4
    patience = 25
    delta = 0.001
    stop_epoch = 50
    num_classes = D.N_CLASSES  # 5
    grad_acc = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    setup_seed(42)

    print("Loading 3-modality data...")
    mods, y, patients, genes = D.load_modalities()
    # Early integration: concatenate cnv + expression + mutation along features.
    X = np.concatenate([mods["cnv"], mods["expression"], mods["mutation"]], axis=1)
    X = X.astype(np.float32)
    print(f"Integrated data shape: {X.shape}  labels: {y.shape}  classes: {num_classes}")

    folds = D.make_folds(y)
    if SMOKE:
        folds = folds[:1]

    per_fold = []
    for fold, (train_idx, val_idx, test_idx) in enumerate(folds):
        print(f"\nFold {fold + 1}/{len(folds)}")

        # Fit scaler on train only (matches original StandardScaler usage).
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx]).astype(np.float32)
        X_val = scaler.transform(X[val_idx]).astype(np.float32)
        X_test = scaler.transform(X[test_idx]).astype(np.float32)
        y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

        train_loader = DataLoader(MultiOmicsDataset(X_train, torch.LongTensor(y_train)),
                                  batch_size=batch_size, shuffle=True,
                                  pin_memory=(device.type == 'cuda'))
        val_loader = DataLoader(MultiOmicsDataset(X_val, torch.LongTensor(y_val)),
                                batch_size=batch_size, pin_memory=(device.type == 'cuda'))
        test_loader = DataLoader(MultiOmicsDataset(X_test, torch.LongTensor(y_test)),
                                 batch_size=batch_size, shuffle=False,
                                 pin_memory=(device.type == 'cuda'))

        model = EarlyIntegrationCNN(in_dim=X.shape[1], num_classes=num_classes)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(device)

        class_counts = np.bincount(y_train, minlength=num_classes)
        class_weight = torch.tensor(
            [sum(class_counts) / (num_classes * x) if x > 0 else 0.0 for x in class_counts],
            dtype=torch.float32).to(device)

        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        loss_fn = nn.CrossEntropyLoss(weight=class_weight)
        early_stopping = EarlyStopping(patience=patience, verbose=False, delta=delta, stop=stop_epoch)

        pin = (device.type == 'cuda')
        for epoch in range(1, epochs + 1):
            model.train()
            for batch_idx, (data, targets) in enumerate(train_loader, start=1):
                data = data.to(device, non_blocking=pin)
                targets = targets.to(device, non_blocking=pin)
                targets = targets[:, 0] if len(targets.shape) > 1 else targets
                outputs = model(data, num_classes=num_classes)
                loss = loss_fn(outputs, targets)
                loss.backward()
                if batch_idx % grad_acc == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e6))
                    optimizer.step()
                    optimizer.zero_grad()

            # Validation: monitor macro-F1 for early stopping.
            model.eval()
            v_prob, v_true = [], []
            with torch.no_grad():
                for data, targets in val_loader:
                    data = data.to(device, non_blocking=pin)
                    targets = targets.to(device, non_blocking=pin)
                    targets = targets[:, 0] if len(targets.shape) > 1 else targets
                    outputs = model(data, num_classes=num_classes)
                    v_prob.append(outputs.detach().cpu().numpy())
                    v_true.append(targets.cpu().numpy())
            v_prob = np.concatenate(v_prob)
            v_true = np.concatenate(v_true)
            f1_macro_val = macro_f1(v_true, v_prob)

            early_stopping([f1_macro_val], epoch)
            if early_stopping.early_stop and epoch > stop_epoch:
                print(f"  Early stopping at epoch {epoch}")
                break

        # Test phase: collect softmax probabilities.
        model.eval()
        t_prob, t_true = [], []
        with torch.no_grad():
            for data, targets in test_loader:
                data = data.to(device, non_blocking=pin)
                targets = targets.to(device, non_blocking=pin)
                targets = targets[:, 0] if len(targets.shape) > 1 else targets
                outputs = model(data, num_classes=num_classes)
                t_prob.append(outputs.detach().cpu().numpy())
                t_true.append(targets.cpu().numpy())
        t_prob = np.concatenate(t_prob)
        t_true = np.concatenate(t_true)

        m = fold_metrics(t_true, t_prob, n_classes=num_classes)
        per_fold.append(m)
        print(f"  Fold {fold + 1} acc={m['accuracy']:.4f} f1={m['f1']:.4f} "
              f"f1_macro={m['f1_macro']:.4f} auc={m['auc']:.4f}")

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    write_metrics_json(RESULTS_JSON, "CNN_ei", "brca_pam50", num_classes, per_fold)
    print(f"\nWrote metrics to {RESULTS_JSON}")


if __name__ == "__main__":
    run()
