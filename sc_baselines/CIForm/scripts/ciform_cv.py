#!/usr/bin/env python3
"""CIForm CV runner — bioMOR baseline adapter.

Thin adapter around the upstream CIForm model (sc_baselines/CIForm/CIForm.py):
reuses its gene->sub-vector embedding (`getXY` logic), `PositionalEncoding`, and the
`CIForm` nn.Module verbatim. We only replace the file/csv data plumbing with
biomor_common's shared X/y arrays + seed-42 CV5 folds, train on train+val, predict on
test per fold, and write the common scores CSV via bc.write_scores.

Ref: Xu et al., "CIForm as a Transformer-based model for cell-type annotation" (Brief Bioinform 2023).
"""
from __future__ import annotations
import argparse, os, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")   # upstream getXY z-scores all-zero sub-vectors -> noisy sklearn warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn import preprocessing
from torch.utils.data import DataLoader, TensorDataset

# --- make repo root + upstream CIForm importable ---
REPO = Path(__file__).resolve().parents[3]          # bioMOR-baselines/
CIFORM_DIR = Path(__file__).resolve().parents[1]     # sc_baselines/CIForm/
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(CIFORM_DIR))
import biomor_common as bc

# Import the upstream model + positional encoding UNCHANGED.
# CIForm.py runs a demo at import (bottom of file), so we import the classes by
# exec'ing only the class/def defs would be fragile; instead we replicate the two
# tiny reusable pieces here to avoid the demo side effects. These are byte-identical
# to CIForm.py's PositionalEncoding / CIForm classes.
import math


def same_seeds(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class CIForm(nn.Module):
    def __init__(self, input_dim, nhead=2, d_model=80, num_classes=2, dropout=0.1):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, dim_feedforward=1024, nhead=nhead, dropout=dropout)
        self.positionalEncoding = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, num_classes))

    def forward(self, mels):
        out = mels.permute(1, 0, 2)
        out = self.positionalEncoding(out)
        out = self.encoder_layer(out)
        out = out.transpose(0, 1)
        out = out.mean(dim=1)
        return self.pred_layer(out)


def build_subvectors(X, gap):
    """CIForm getXY sub-vector embedding: (N,G) -> (N, ceil(G/gap), gap), each
    sub-vector z-scored (preprocessing.scale). Verbatim to upstream getXY inner loop."""
    N, G = X.shape
    feats = []
    for cell in X:
        length = len(cell)
        sub = []
        for k in range(0, length, gap):
            a = cell[k:k + gap] if (k + gap <= length) else cell[length - gap:length]
            a = preprocessing.scale(a)
            sub.append(a)
        feats.append(np.asarray(sub, dtype=np.float32))
    return np.asarray(feats, dtype=np.float32)


def hvg_idx(Xlog, topgenes):
    """Seurat-style dispersion HVG on log data, train-fold only. Returns gene indices."""
    if Xlog.shape[1] <= topgenes:
        return np.arange(Xlog.shape[1])
    mean = Xlog.mean(0)
    var = Xlog.var(0)
    disp = np.divide(var, mean, out=np.zeros_like(var), where=mean > 0)
    return np.argsort(-disp)[:topgenes]


def run_fold(X, y, tr, va, te, gap, topgenes, epochs, batch_size, lr, dropout,
             nhead, device):
    tr_all = np.concatenate([tr, va])
    Xtr, ytr = X[tr_all], y[tr_all]
    Xte, yte = X[te], y[te]

    # log1p (matches CIForm sc.pp.log1p), then HVG on train fold, apply to both.
    Xtr = np.log1p(Xtr); Xte = np.log1p(Xte)
    gi = hvg_idx(Xtr, topgenes)
    Xtr, Xte = Xtr[:, gi], Xte[:, gi]

    d_model = gap
    Ztr = build_subvectors(Xtr, gap)
    Zte = build_subvectors(Xte, gap)

    num_classes = int(y.max()) + 1
    same_seeds(2021)
    model = CIForm(input_dim=d_model, nhead=nhead, d_model=d_model,
                   num_classes=num_classes, dropout=dropout).to(device)
    model = model.float()
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    tr_ds = TensorDataset(torch.from_numpy(Ztr).float(),
                          torch.from_numpy(ytr).long())
    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, pin_memory=True)

    model.train()
    for ep in range(epochs):
        tot = 0.0
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = crit(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        print(f"    epoch {ep+1}/{epochs} loss={tot/len(tr_ld):.4f}", flush=True)

    model.eval()
    preds = []
    te_ld = DataLoader(TensorDataset(torch.from_numpy(Zte).float()),
                       batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (xb,) in te_ld:
            preds.append(model(xb.to(device)).argmax(1).cpu().numpy())
    yp = np.concatenate(preds)
    f1, acc = bc.fold_metrics(yte, yp)
    return f1, acc, len(te)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--gap", type=int, default=1024)
    ap.add_argument("--topgenes", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--work_dir", default=None)
    args = ap.parse_args()

    # gap must be divisible by nhead (d_model=gap). Adjust nhead if needed.
    if args.gap % args.nhead != 0:
        for h in [8, 4, 2, 1]:
            if args.gap % h == 0:
                args.nhead = h; break

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CIForm] dataset={args.dataset} device={device} gap={args.gap} "
          f"nhead={args.nhead} epochs={args.epochs}", flush=True)

    X, y, genes = bc.load_sc(args.dataset)
    print(f"  loaded X{X.shape} classes={int(y.max())+1}", flush=True)
    folds = bc.load_sc_folds(args.dataset, y)

    f1s, accs, nts = [], [], []
    for i, (tr, va, te) in enumerate(folds[:args.folds]):
        print(f"  fold {i+1}/{min(args.folds,len(folds))}", flush=True)
        f1, acc, nt = run_fold(X, y, tr, va, te, args.gap, args.topgenes,
                               args.epochs, args.batch_size, args.lr,
                               args.dropout, args.nhead, device)
        print(f"  fold {i+1} macro_f1={f1:.2f} acc={acc:.2f}", flush=True)
        f1s.append(f1); accs.append(acc); nts.append(nt)

    wd = args.work_dir or str(CIFORM_DIR / "work_dirs" / args.dataset)
    out = bc.write_scores(wd, "CIForm", args.dataset, f1s, accs, nts)
    print(f"[CIForm] wrote {out}", flush=True)
    print(f"[CIForm] mean macro_f1={np.mean(f1s):.2f} acc={np.mean(accs):.2f}", flush=True)


if __name__ == "__main__":
    main()
