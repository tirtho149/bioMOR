"""Deep / foundation-model baselines for the genoNet classification tasks.

APPLE-TO-APPLE by construction: every baseline is evaluated on the SAME data,
the SAME stratified 80/20 split (seed 42) and the SAME metrics
(accuracy / macro-F1 / weighted-F1) as ``baselines.py`` and ``genonet_tasks.py``
(unified_bio5.csv, all 20530 genes). Results are written one JSON per task to
``results_dl_baselines/<task>.json`` with the identical schema used by
``baselines.py``, so ``make_paper.foundation_perf_table`` can drop them straight
into the paper next to the classical-baseline table.

The seven reproducible baselines (clones live in
``lit_pipeline/baseline_repos/<name>``; checkpoints already fetched where needed):

  ready, no GPU / no weights:
    scGeneFit         marker selection (LP) + NearestCentroid   [FULLY IMPLEMENTED]
    sciLaMA           VAE embedding + logistic-regression probe  [adapter stub]
    scbenchmark       6-layer/256-d transformer (their baseline) [adapter stub]
  ready, GPU + fetched checkpoint:
    scGPT             pan-cancer ckpt, fine-tune cls head        [adapter stub]
    Geneformer        V2-104M_CLcancer, fine-tune cls head       [adapter stub]
    CellPLM           20231027_85M, fine-tune cls head           [adapter stub]
    Cell2Sentence     C2S-Pythia-410M, cell-sentence + LLM head  [adapter stub]

Stubs raise NotImplementedError with a one-line pointer to the upstream repo's
training entrypoint; each is meant to be run inside that repo's own conda env
(their CUDA/torch/flash-attn pins differ), driven by run_dl_baselines.sbatch.
scGeneFit runs in THIS venv with no GPU.

Usage:
    python -m recursive_marker_transformer.dl_baselines --params           # write params.json
    python -m recursive_marker_transformer.dl_baselines --only scGeneFit   # run one baseline (CPU)
    python -m recursive_marker_transformer.dl_baselines --only scGPT --tasks pathologic_stage
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from .genonet_tasks import TASKS, _load_unified

SEED = 42
_REPO = Path(__file__).resolve().parents[1]
_REPOS = _REPO / "lit_pipeline" / "baseline_repos"
OUT = _REPO / "results_dl_baselines"


# --------------------------------------------------------------------------- #
# shared split + metrics (identical to baselines.py)                           #
# --------------------------------------------------------------------------- #
def _split(X, labels, task, seed=SEED):
    y_raw = labels[task].values
    remap = {v: i for i, v in enumerate(np.unique(y_raw))}
    y = np.array([remap[v] for v in y_raw], dtype=np.int64)
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y)
    mu, sd = X[tr].mean(0, keepdims=True), X[tr].std(0, keepdims=True) + 1e-6
    Xs = (X - mu) / sd
    return Xs[tr], Xs[te], y[tr], y[te]


def _metrics(yte, yp, t0):
    return {
        "accuracy": float(accuracy_score(yte, yp)),
        "macro_f1": float(f1_score(yte, yp, average="macro")),
        "weighted_f1": float(f1_score(yte, yp, average="weighted")),
        "seconds": round(time.time() - t0, 1),
    }


# --------------------------------------------------------------------------- #
# adapters: each returns predictions on the test split                         #
# --------------------------------------------------------------------------- #
def run_scgenefit(Xtr, Xte, ytr, yte, n_markers=256):
    """scGeneFit: supervised LP marker selection on train, classify with the
    selected markers (NearestCentroid) -- mirrors the repo README example.
    Runs on CPU in this venv (uses the cloned source; the PyPI build pins the
    deprecated `sklearn` stub, so we import from the clone directly)."""
    import sys
    sys.path.insert(0, str(_REPOS / "scGeneFit"))
    from scGeneFit.functions import get_markers
    from sklearn.neighbors import NearestCentroid
    markers = get_markers(Xtr, ytr, n_markers, method="centers", redundancy=0.25, verbose=False)
    clf = NearestCentroid().fit(Xtr[:, markers], ytr)
    return clf.predict(Xte[:, markers])


def run_scilama(Xtr, Xte, ytr, yte):
    """sciLaMA embedding (train VAE) + logistic-probe on the labels.
    Adapter: import SciLaMATrainer from the clone, fit on an AnnData wrapping
    Xtr, read adata.obsm['X_sciLaMA'], then LogisticRegression on the labels."""
    raise NotImplementedError(
        "sciLaMA adapter: build AnnData(Xtr) with obs['split'], "
        f"SciLaMATrainer at {_REPOS/'sciLaMA'}; embed -> LogisticRegression probe.")


def run_scbenchmark(Xtr, Xte, ytr, yte):
    """scbenchmark 6-layer/256-d transformer baseline (train_baseline.py).
    Adapter: format the split into its scGPT-style binned input and call its
    downstream classification head."""
    raise NotImplementedError(
        f"scbenchmark adapter: use {_REPOS/'scbenchmark'}/train_baseline.py "
        "(--model_structure transformer) on the binned split.")


def _gpu_stub(name, entry):
    def _fn(Xtr, Xte, ytr, yte):
        raise NotImplementedError(
            f"{name} adapter: run inside the repo's own conda env via {entry}. "
            "Match genes to its vocab, fine-tune the cls head on (Xtr,ytr), predict Xte. "
            "Driven by run_dl_baselines.sbatch.")
    return _fn


ADAPTERS = {
    "scGeneFit":     run_scgenefit,                       # CPU, ready
    "sciLaMA":       run_scilama,
    "scbenchmark":   run_scbenchmark,
    "scGPT":         _gpu_stub("scGPT", "scGPT/checkpoints/pan-cancer + examples/finetune_integration.py"),
    "Geneformer":    _gpu_stub("Geneformer", "Geneformer-V2-104M_CLcancer + examples/cell_classification"),
    "CellPLM":       _gpu_stub("CellPLM", "CellPLM/ckpt/ckpt/20231027_85M + tutorials/cell_type_annotation"),
    "Cell2Sentence": _gpu_stub("Cell2Sentence", "C2S-Pythia-410M + tutorial 4 cell-type prediction"),
}


# --------------------------------------------------------------------------- #
def run_one(name, X, labels, tasks):
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{name}.json"
    res = json.loads(out.read_text()) if out.exists() else {"method": name, "tasks": {}}
    for task in tasks:
        Xtr, Xte, ytr, yte = _split(X, labels, task)
        t0 = time.time()
        try:
            yp = ADAPTERS[name](Xtr, Xte, ytr, yte)
            res["tasks"][task] = _metrics(yte, yp, t0)
            m = res["tasks"][task]
            print(f"  [{name}] {task:18s} acc={m['accuracy']*100:5.1f} "
                  f"macroF1={m['macro_f1']*100:5.1f} ({m['seconds']}s)", flush=True)
        except NotImplementedError as e:
            print(f"  [{name}] {task:18s} STUB: {e}", flush=True)
        except Exception as e:
            res["tasks"][task] = {"error": str(e)}
            print(f"  [{name}] {task:18s} ERROR {e}", flush=True)
        out.write_text(json.dumps(res, indent=1))
    return res


def write_params():
    """params.json already holds the real (checkpoint-measured) counts; this
    just re-validates it exists so --params is a no-op safety check."""
    p = OUT / "params.json"
    if p.exists():
        print(f"[params] {p} present ({len(json.loads(p.read_text())['baselines'])} baselines)")
    else:
        print(f"[params] MISSING {p} -- regenerate from checkpoints.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=_REPO / "data/tcga/unified_bio5.csv")
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--only", nargs="*", default=list(ADAPTERS), help="baselines to run")
    ap.add_argument("--params", action="store_true", help="just validate params.json")
    args = ap.parse_args()

    if args.params:
        write_params(); return

    print(f"[dl_baselines] loading {args.csv} ...", flush=True)
    X, labels, gene_cols = _load_unified(args.csv)
    print(f"[dl_baselines] X={X.shape} genes={len(gene_cols)} tasks={args.tasks}", flush=True)
    for name in args.only:
        print(f"\n===== {name} =====", flush=True)
        run_one(name, X, labels, args.tasks)
    print("\n[dl_baselines] done -> " + str(OUT), flush=True)


if __name__ == "__main__":
    main()
