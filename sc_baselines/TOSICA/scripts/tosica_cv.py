#!/usr/bin/env python3
"""TOSICA CV runner — bioMOR baseline adapter.

Thin adapter around upstream TOSICA (sc_baselines/TOSICA/TOSICA/): reuses the
`scTrans_model` transformer (TOSICA_model.py) and the pathway-mask machinery
verbatim. We replace TOSICA.train/TOSICA.pre (which are AnnData+file/attention
oriented) with a plain train/predict loop over biomor_common's shared X/y arrays
and seed-42 CV5 folds; train on train+val, predict test per fold; write the
common scores CSV via bc.write_scores.

No .gmt resources are bundled in this checkout, so we use TOSICA's own
gmt_path=None fallback: a random-binomial pathway mask (fully-connected tokens).
This is the model author's supported "Full connection!" path and keeps the
architecture identical; it just drops the pathway-interpretability of the tokens.

Ref: Chen et al., "Transformer for one stop interpretable cell type annotation"
(Nat Commun 2023).
"""
from __future__ import annotations
import argparse, importlib.util, math, os, random, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parents[3]
TOSICA_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import biomor_common as bc

# Load the upstream model module directly by path, bypassing TOSICA/__init__.py
# (which pulls in train.py -> torch.utils.tensorboard -> tensorboard, not installed).
# We first register the customized_linear dependency under the package name that
# TOSICA_model.py's relative import expects.
_PKG = TOSICA_DIR / "TOSICA"
import types
_pkg_mod = types.ModuleType("TOSICA")
_pkg_mod.__path__ = [str(_PKG)]
sys.modules["TOSICA"] = _pkg_mod
for _name in ("customized_linear", "TOSICA_model"):
    _spec = importlib.util.spec_from_file_location(
        f"TOSICA.{_name}", str(_PKG / f"{_name}.py"))
    _m = importlib.util.module_from_spec(_spec)
    sys.modules[f"TOSICA.{_name}"] = _m
    _spec.loader.exec_module(_m)
create_model = sys.modules["TOSICA.TOSICA_model"].scTrans_model  # upstream, unchanged


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def make_mask(n_genes, max_gs, mask_ratio, seed):
    """TOSICA gmt_path=None fallback: random-binomial gene->token mask."""
    rng = np.random.RandomState(seed)
    mask = rng.binomial(1, mask_ratio, size=(n_genes, max_gs)).astype(np.float32)
    # guard: every token must connect >=1 gene, every gene >=1 token, so the
    # CustomizedLinear has no dead columns/rows.
    for j in range(max_gs):
        if mask[:, j].sum() == 0:
            mask[rng.randint(n_genes), j] = 1.0
    dead = np.where(mask.sum(1) == 0)[0]
    for i in dead:
        mask[i, rng.randint(max_gs)] = 1.0
    return mask


def run_fold(X, y, tr, va, te, max_gs, mask_ratio, embed_dim, depth, num_heads,
             epochs, batch_size, lr, lrf, device, seed=1):
    tr_all = np.concatenate([tr, va])
    Xtr, ytr = X[tr_all].astype(np.float32), y[tr_all]
    Xte, yte = X[te].astype(np.float32), y[te]

    n_genes = X.shape[1]
    num_classes = int(y.max()) + 1

    set_seed(seed)
    mask = make_mask(n_genes, max_gs, mask_ratio, seed)
    model = create_model(num_classes=num_classes, num_genes=n_genes, mask=mask,
                         embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                         has_logits=False).to(device)

    crit = nn.CrossEntropyLoss()
    pg = [p for p in model.parameters() if p.requires_grad]
    opt = optim.SGD(pg, lr=lr, momentum=0.9, weight_decay=5e-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf
    sched = lr_scheduler.LambdaLR(opt, lr_lambda=lf)

    tr_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr).long())
    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                       pin_memory=True, drop_last=True)

    model.train()
    for ep in range(epochs):
        tot = 0.0; nb = 0
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            _, pred, _ = model(xb)
            loss = crit(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        print(f"    epoch {ep+1}/{epochs} loss={tot/max(nb,1):.4f}", flush=True)

    model.eval()
    preds = []
    te_ld = DataLoader(TensorDataset(torch.from_numpy(Xte)),
                       batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (xb,) in te_ld:
            _, pred, _ = model(xb.to(device))
            preds.append(pred.argmax(1).cpu().numpy())
    yp = np.concatenate(preds)
    f1, acc = bc.fold_metrics(yte, yp)
    return f1, acc, len(te)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--max_gs", type=int, default=300)     # number of pathway tokens
    ap.add_argument("--mask_ratio", type=float, default=0.015)
    ap.add_argument("--embed_dim", type=int, default=48)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--lrf", type=float, default=0.01)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--work_dir", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TOSICA] dataset={args.dataset} device={device} max_gs={args.max_gs} "
          f"embed_dim={args.embed_dim} depth={args.depth} epochs={args.epochs}",
          flush=True)

    X, y, genes = bc.load_sc(args.dataset)
    print(f"  loaded X{X.shape} classes={int(y.max())+1}", flush=True)
    folds = bc.load_sc_folds(args.dataset, y)

    f1s, accs, nts = [], [], []
    for i, (tr, va, te) in enumerate(folds[:args.folds]):
        print(f"  fold {i+1}/{min(args.folds,len(folds))}", flush=True)
        f1, acc, nt = run_fold(X, y, tr, va, te, args.max_gs, args.mask_ratio,
                               args.embed_dim, args.depth, args.num_heads,
                               args.epochs, args.batch_size, args.lr, args.lrf,
                               device)
        print(f"  fold {i+1} macro_f1={f1:.2f} acc={acc:.2f}", flush=True)
        f1s.append(f1); accs.append(acc); nts.append(nt)

    wd = args.work_dir or str(TOSICA_DIR / "work_dirs" / args.dataset)
    out = bc.write_scores(wd, "TOSICA", args.dataset, f1s, accs, nts)
    print(f"[TOSICA] wrote {out}", flush=True)
    print(f"[TOSICA] mean macro_f1={np.mean(f1s):.2f} acc={np.mean(accs):.2f}", flush=True)


if __name__ == "__main__":
    main()
