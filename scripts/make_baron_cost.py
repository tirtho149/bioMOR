"""Timed per-epoch training on Baron for the four architectures (Table-2 configs), to build
the compute-cost figures: learning curves (F1 vs cumulative GPU-seconds) and time-to-target.

Records, per epoch, {epoch, train_loss, val_f1, sec} for Vanilla / Recursive / MoR / bioMoR
on the same fold-0 of the shared 5-fold split, plus final test F1 and parameter count.
"""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on path (script lives in scripts/)
from recursive_marker_transformer.singlecell import _load_dataset, _fit_eval, HEAD
from recursive_marker_transformer.config import RMTConfig
from recursive_marker_transformer.cv import cv_folds, SEED, VAL_FRAC
from recursive_marker_transformer.bio_learned_genomap import _cfg as bio_cfg

ROOT = Path(__file__).resolve().parent.parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def sc_cfg(mode, share):
    return RMTConfig(heads=(HEAD,), n_hvg=None, batch_size=128, d_model=96, d_ff=192,
                     n_markers=128, marker_mode="router", recursion_mode=mode,
                     recursion_depth=4, share_weights=share, seed=SEED,
                     epochs=100, patience=15, lr=1e-3, weight_decay=1e-5)


def configs():
    return {
        "Vanilla":   sc_cfg("expert", False),   # K independent layers (no weight sharing)
        "Recursive": sc_cfg("fixed",  True),    # weight-shared, fixed depth
        "MoR":       sc_cfg("expert", True),    # weight-shared adaptive (Mixture-of-Recursions)
        "bioMoR":    bio_cfg("learned", 4, SEED, 100, n_markers=128),
    }


def main(smoke=False):
    X, y, _ = _load_dataset(ROOT / "data" / "singlecell" / "baron")
    X = X.astype(np.float32); F = X.shape[1]; C = int(y.max() + 1)
    tr, va, te = list(cv_folds(y, n_folds=5, seed=SEED, val_frac=VAL_FRAC))[0]
    out = {}
    for name, cfg in configs().items():
        if smoke:
            cfg.epochs = 2
        cfg.n_markers = min(cfg.n_markers, F)
        torch.manual_seed(SEED); np.random.seed(SEED)
        yt, yp, model = _fit_eval(X, y, tr, va, te, cfg, F, C, DEV)
        h = getattr(model, "_history", [])
        tf1 = 100.0 * f1_score(yt, yp, average="macro")
        out[name] = {"history": h, "test_f1": tf1,
                     "params": int(sum(p.numel() for p in model.parameters()))}
        last = h[-1] if h else {"sec": 0}
        print(f"[baron-cost] {name}: epochs={len(h)} sec={last['sec']:.1f} "
              f"test_f1={tf1:.1f} params={out[name]['params']}", flush=True)
    d = ROOT / "results/cv5" / "curves"; d.mkdir(parents=True, exist_ok=True)
    (d / "baron_cost.json").write_text(json.dumps(out, indent=1))
    print(f"[baron-cost] saved {d/'baron_cost.json'}", flush=True)


if __name__ == "__main__":
    import sys
    main(smoke="--smoke" in sys.argv)
