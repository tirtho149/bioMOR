"""Extract per-pathway recursion depth for prostate (multi-omics) under expert-choice
vs token-choice routing, with real Reactome pathway names -> results/depth/prostate_panels.json.
Feeds the Figure-2 right panels (genes/pathways flowing to their exit depth)."""
import json, numpy as np, torch, pandas as pd
from dataclasses import replace
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on path (script lives in scripts/)
from recursive_marker_transformer.config import RMTConfig
from recursive_marker_transformer.pathway_data import load_cohort, _first_existing
from recursive_marker_transformer.pathway_tasks import _fit_eval
from recursive_marker_transformer.cv import cv_folds, SEED, VAL_FRAC
from recursive_marker_transformer.train import _depth_stats, resolve_device
from pathlib import Path

coh = load_cohort("prostate", channels="mut_cnv")
X, y = coh.X, coh.y
G = X.shape[1]; K = int(y.max() + 1); C = 1 if X.ndim == 2 else X.shape[2]
dtypes = {"prostate": "multiclass"}
pwf = _first_existing(Path("data/prostate"), "filtered_pathways.csv", "pathways.csv")
pw = pd.read_csv(pwf); id2name = dict(zip(pw["Pathway_ID"], pw["Pathway_Name"]))
names = [str(id2name.get(pid, pid)) for pid in coh.pathways]
device = resolve_device("cuda")
tr, va, te = list(cv_folds(y, n_folds=5, seed=SEED, val_frac=VAL_FRAC))[0]

out = {"pathways": names, "recursion_depth": 4}
for mode in ["expert", "token"]:
    cfg = replace(RMTConfig(), heads=("prostate",), n_hvg=None, n_channels=C,
                  d_model=128, d_ff=256, marker_mode="pathway", recursion_mode=mode,
                  gene_interaction="reactome", share_weights=True, recursion_depth=4,
                  epochs=100, patience=15, batch_size=32, lr=3e-4, seed=42)
    yt, yp, model, dl_te, _ = _fit_eval("prostate", coh, X, y, tr, va, te, cfg, G, K, dtypes, device)
    md, midx, active = _depth_stats(model, dl_te, device, cfg)
    md = md.numpy()
    out[mode] = {"depths": md.tolist(),
                 "active_per_step": [float(a) / len(md) for a in active.tolist()]}
    print(f"[panels] {mode}: mean_depth={md.mean():.2f} active/step={out[mode]['active_per_step']}", flush=True)
Path("results/depth").mkdir(exist_ok=True)
json.dump(out, open("results/depth/prostate_panels.json", "w"))
print("wrote results/depth/prostate_panels.json")
