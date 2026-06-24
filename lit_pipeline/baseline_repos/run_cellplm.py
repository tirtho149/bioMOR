#!/usr/bin/env python3
"""CellPLM baseline: zero-shot cell embedding (85M ckpt) + logistic-regression
probe on labels. SAME seed-42 split as SMART. Writes results_dl_baselines/CellPLM.json.
Run inside the cellplm env."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path("/work/mech-ai-scratch/tirtho/RecusrsiveQFormer")
sys.path.insert(0, str(REPO / "lit_pipeline/baseline_repos/CellPLM"))
GF = REPO / "lit_pipeline/baseline_repos/Geneformer"
AUX = REPO / "data/geneformer_aux"
OUT = REPO / "results_dl_baselines"
CKPT_DIR = REPO / "lit_pipeline/baseline_repos/CellPLM/ckpt/ckpt"
PREFIX = "20231027_85M"
TASKS = ["pathologic_stage", "pathologic_T", "pathologic_N", "os_binary", "tumor_status"]
SEED = 42


def to_ensembl(gene_cols):
    import pickle, json as _j
    name2id = pickle.load(open(GF / "geneformer/gene_name_id_dict_gc104M.pkl", "rb"))
    e2ens = _j.load(open(AUX / "entrez2ensembl_human.json"))
    cols, ens = [], []
    for c in gene_cols:
        sym, ent = (c.split("|", 1) + [None])[:2] if "|" in c else (c, None)
        g = name2id.get(sym) if sym and sym != "?" else None
        if g is None and ent:
            g = e2ens.get(ent)
        if g:
            cols.append(c); ens.append(g)
    return cols, ens


def run_task(task):
    import anndata as ad, torch
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from CellPLM.pipeline.cell_embedding import CellEmbeddingPipeline

    df = pd.read_csv(REPO / "data/tcga/unified_bio5.csv")
    meta = {"sample","cancer_type","cancer_name","os_binary","pathologic_stage",
            "pathologic_T","pathologic_N","tumor_status"}
    gcols = [c for c in df.columns if c not in meta]
    df = df.dropna(subset=[task]).reset_index(drop=True)
    cols, ens = to_ensembl(gcols)
    # de-dup ensembl (keep first)
    seen, kc, ke = set(), [], []
    for c, e in zip(cols, ens):
        if e not in seen:
            seen.add(e); kc.append(c); ke.append(e)
    X = df[kc].to_numpy(np.float32)
    y = df[task].astype(str).values
    a = ad.AnnData(X=X, obs=pd.DataFrame({"label": y}))
    a.var_names = ke
    print(f"[{task}] genes->ensembl {len(ke)}/{len(gcols)}; N={len(df)}", flush=True)

    pipe = CellEmbeddingPipeline(pretrain_prefix=PREFIX, pretrain_directory=str(CKPT_DIR))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    emb = pipe.predict(a, device=dev)
    emb = emb.cpu().numpy() if hasattr(emb, "cpu") else np.asarray(emb)

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y)
    clf = LogisticRegression(max_iter=2000).fit(emb[tr], y[tr])
    yp = clf.predict(emb[te])
    res = {"accuracy": float(accuracy_score(y[te], yp)),
           "macro_f1": float(f1_score(y[te], yp, average="macro")),
           "weighted_f1": float(f1_score(y[te], yp, average="weighted")),
           "n_test": len(te), "n_genes_used": len(ke)}
    print(f"[{task}] acc={res['accuracy']*100:.1f} macroF1={res['macro_f1']*100:.1f}", flush=True)
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "CellPLM.json"
    res = json.loads(out.read_text()) if out.exists() else {"method": "CellPLM", "tasks": {}}
    for t in TASKS:
        try:
            res["tasks"][t] = run_task(t)
        except Exception as e:
            import traceback; traceback.print_exc(); res["tasks"][t] = {"error": str(e)}
        out.write_text(json.dumps(res, indent=1))
    print("[cellplm] done ->", out)


if __name__ == "__main__":
    main()
