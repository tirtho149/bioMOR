#!/usr/bin/env python3
"""sciLaMA baseline: train its beta-VAE cell representation on the genoNet
expression (no external LLM embeddings), then a logistic-regression probe on the
labels. SAME seed-42 split as SMART. Writes results_dl_baselines/sciLaMA.json.
Run inside the sciLaMA env."""
import sys, json, yaml, shutil
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path("/work/mech-ai-scratch/tirtho/RecusrsiveQFormer")
sys.path.insert(0, str(REPO / "lit_pipeline/baseline_repos/sciLaMA/src"))
OUT = REPO / "results_dl_baselines"
TASKS = ["pathologic_stage", "pathologic_T", "pathologic_N", "os_binary", "tumor_status"]
SEED = 42


def run_task(task, work):
    import anndata as ad, scanpy as sc
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sciLaMA.trainer import SciLaMATrainer

    df = pd.read_csv(REPO / "data/tcga/unified_bio5.csv")
    meta = {"sample","cancer_type","cancer_name","os_binary","pathologic_stage",
            "pathologic_T","pathologic_N","tumor_status"}
    gcols = [c for c in df.columns if c not in meta]
    df = df.dropna(subset=[task]).reset_index(drop=True)
    X = df[gcols].to_numpy(np.float32)
    y = df[task].astype(str).values
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y)
    trn, val = train_test_split(tr, test_size=0.15, random_state=SEED, stratify=y[tr])
    mu, sd = X[trn].mean(0, keepdims=True), X[trn].std(0, keepdims=True) + 1e-6
    Xs = (X - mu) / sd
    split = np.array(["train"] * len(y), dtype=object)
    split[val] = "val"; split[te] = "test"
    a = ad.AnnData(X=Xs, obs=pd.DataFrame({"label": y, "split": split}))
    a.var_names = [c.replace("|", "_") for c in gcols]
    wd = Path(work); wd.mkdir(parents=True, exist_ok=True)
    h5 = wd / "data.h5ad"; a.write_h5ad(h5)

    cfg = {
        "data": {"path": str(h5), "check_scaling": False, "split_column": "split",
                 "train_split_key": "train", "val_split_key": "val", "test_split_key": "test",
                 "categorical_covariate_keys": None, "continuous_covariate_keys": None},
        "model": {"hidden_dims": [900, 400], "latent_dim": 50, "dropout_rate": 0.1,
                  "batchnorm": False, "layernorm": True, "activation": "LeakyReLU", "var_eps": 1e-4},
        "training": {"seed": SEED, "mode": "beta_vae", "max_epochs": 200, "batch_size": 128,
                     "devices": 1, "strategy": "auto", "learning_rate": 1e-3, "patience": 20,
                     "weight_decay": 0.0, "beta_start": 0.0, "beta_end": 1.0,
                     "epochs_before_beta_warmup": 25, "beta_warmup_rate": 0.05, "gamma": 0.05},
        "output": {"save_dir": str(wd / "out"), "save_key": "X_sciLaMA"},
    }
    cfgp = wd / "cfg.yaml"; cfgp.write_text(yaml.dump(cfg))
    tr_ = SciLaMATrainer(str(cfgp)); tr_.train()
    emb = tr_.datamodule.adata.obsm["X_sciLaMA"]
    obs = tr_.datamodule.adata.obs
    is_tr = (obs["split"] != "test").values
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(emb[is_tr], obs["label"].values[is_tr])
    yp = clf.predict(emb[~is_tr]); gt = obs["label"].values[~is_tr]
    res = {"accuracy": float(accuracy_score(gt, yp)),
           "macro_f1": float(f1_score(gt, yp, average="macro")),
           "weighted_f1": float(f1_score(gt, yp, average="weighted")),
           "n_test": int((~is_tr).sum())}
    print(f"[{task}] acc={res['accuracy']*100:.1f} macroF1={res['macro_f1']*100:.1f}", flush=True)
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "sciLaMA.json"
    res = json.loads(out.read_text()) if out.exists() else {"method": "sciLaMA", "tasks": {}}
    for t in TASKS:
        try:
            res["tasks"][t] = run_task(t, REPO / "results_dl_baselines/_scilama_work" / t)
        except Exception as e:
            import traceback; traceback.print_exc(); res["tasks"][t] = {"error": str(e)}
        out.write_text(json.dumps(res, indent=1))
    print("[scilama] done ->", out)


if __name__ == "__main__":
    main()
