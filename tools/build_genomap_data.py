#!/usr/bin/env python3
"""Convert data-branch `genomap_data/` datasets into the `data/singlecell/<name>/`
format the trainer expects (expression.csv.gz + labels.csv).

Adds the new *supervised* genomap datasets to the single-cell suite:
  - Ischaemic: Lung, Oesophagus, Spleen   ({name}_data.npy [N,1089] + GT_{name}.mat)
  - Tcell:     Elyahu2019_SCP490          (tcell_data.npy [N,1089] + GT_tcell.mat)

Each source ships a {name}_data.npy already reduced to the 1089 genomap HVGs (so the
feature dim matches tabula_muris/common_class), an integer label vector in
GT_{name}.mat (key 'GT', 1-based), and a labelmap.json (0-based idx -> class name).

Usage:
  python tools/build_genomap_data.py                 # build all available
  python tools/build_genomap_data.py --only lung tcell
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
GD = ROOT / "genomap_data"
OUT = ROOT / "data" / "singlecell"

# name -> (data .npy, GT .mat, labelmap .json)
SOURCES = {
    "lung":       ("Ischaemic/Lung/lung_data.npy",             "Ischaemic/Lung/GT_lung.mat",             "Ischaemic/Lung/lung_labelmap.json"),
    "oesophagus": ("Ischaemic/Oesophagus/oesophagus_data.npy", "Ischaemic/Oesophagus/GT_oesophagus.mat", "Ischaemic/Oesophagus/oesophagus_labelmap.json"),
    "spleen":     ("Ischaemic/Spleen/spleen_data.npy",         "Ischaemic/Spleen/GT_spleen.mat",         "Ischaemic/Spleen/spleen_labelmap.json"),
    "tcell":      ("Tcell/Elyahu2019_SCP490/tcell_data.npy",   "Tcell/Elyahu2019_SCP490/GT_tcell.mat",   "Tcell/Elyahu2019_SCP490/tcell_labelmap.json"),
}


def _is_lfs_pointer(p: Path) -> bool:
    try:
        return p.read_bytes()[:60].startswith(b"version https://git-lfs")
    except Exception:
        return True


def _load_gt(mat_path: Path) -> np.ndarray:
    m = sio.loadmat(mat_path)
    key = next(k for k in m if not k.startswith("__"))
    return np.asarray(m[key]).ravel().astype(np.int64)


def build_one(name: str) -> dict:
    npy, gt, lm = (GD / s for s in SOURCES[name])
    for f in (npy, gt, lm):
        if not f.exists() or _is_lfs_pointer(f):
            raise FileNotFoundError(f"{f} missing or still an LFS pointer (run git lfs pull)")

    X = np.load(npy, allow_pickle=True).astype(np.float32)
    y = _load_gt(gt)                              # 1-based integer labels
    labelmap = json.load(open(lm))               # {"0": "name", ...} 0-based
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"{name}: X rows {X.shape[0]} != labels {y.shape[0]}")

    N, Fdim = X.shape
    cell_ids = [f"{name}_{i:06d}" for i in range(N)]
    names = [labelmap.get(str(int(v) - 1), str(int(v))) for v in y]   # GT 1-based -> labelmap 0-based

    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    cols = [f"gene_{j:04d}" for j in range(Fdim)]
    expr = pd.DataFrame(X, index=pd.Index(cell_ids, name="cell_id"), columns=cols)
    expr.to_csv(out / "expression.csv.gz", compression="gzip")
    pd.DataFrame({"cell_id": cell_ids, "label": y.astype(int), "class_name": names}) \
        .to_csv(out / "labels.csv", index=False)
    return {"dataset": name, "n_samples": int(N), "n_features": int(Fdim),
            "n_classes": int(len(np.unique(y))), "has_class_names": True, "split": "none"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="subset of dataset names")
    args = ap.parse_args()
    names = args.only or list(SOURCES)
    built = []
    for n in names:
        try:
            info = build_one(n)
            built.append(info)
            print(f"[build] {n}: {info['n_samples']} cells x {info['n_features']} genes, "
                  f"{info['n_classes']} classes -> {OUT / n}")
        except Exception as e:
            print(f"[skip] {n}: {e}")
    if built:
        print(f"[done] built {len(built)}/{len(names)} datasets: "
              f"{', '.join(b['dataset'] for b in built)}")


if __name__ == "__main__":
    main()
