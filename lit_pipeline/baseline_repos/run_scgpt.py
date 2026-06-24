#!/usr/bin/env python3
"""scGPT baseline: frozen pan-cancer cell embedding + logistic-regression probe
on labels. SAME seed-42 split as SMART. Writes results_dl_baselines/scGPT.json.
Run inside the scgpt env.

Notes / documented approximations:
  * uses the LOCAL repo copy of scgpt (torchtext-free vocab_compat shim), not the
    pip wheel 0.2.4 which hard-imports torchtext and breaks on torch 2.11;
  * flash-attn is not installed -> use_fast_transformer=False;
  * gene id: column 'SYMBOL|entrez' -> SYMBOL, matched against scGPT's gene-symbol
    vocab (genes absent from the vocab are dropped by embed_data);
  * input values are log2(x+1) RSEM (bulk), which scGPT rank-bins internally;
    bulk-on-a-single-cell-model is OOD, reported transparently in the paper caption.
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path("/work/mech-ai-scratch/tirtho/RecusrsiveQFormer")
# local repo scgpt FIRST, so the torchtext-free copy shadows the pip wheel
sys.path.insert(0, str(REPO / "lit_pipeline/baseline_repos/scGPT"))
OUT = REPO / "results_dl_baselines"
CKPT = REPO / "lit_pipeline/baseline_repos/scGPT/checkpoints/pan-cancer"
TASKS = ["pathologic_stage", "pathologic_T", "pathologic_N", "os_binary", "tumor_status"]
SEED = 42


def run_task(task):
    import anndata as ad, torch
    from scgpt.tasks import embed_data
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    df = pd.read_csv(REPO / "data/tcga/unified_bio5.csv")
    meta = {"sample","cancer_type","cancer_name","os_binary","pathologic_stage",
            "pathologic_T","pathologic_N","tumor_status"}
    gcols = [c for c in df.columns if c not in meta]
    df = df.dropna(subset=[task]).reset_index(drop=True)

    # gene symbol (strip |entrez), drop unknown '?' and dedup keeping first
    syms, keep = [], []
    seen = set()
    for c in gcols:
        s = c.split("|", 1)[0]
        if s and s != "?" and s not in seen:
            seen.add(s); keep.append(c); syms.append(s)
    X = df[keep].to_numpy(np.float32)
    y = df[task].astype(str).values
    a = ad.AnnData(X=X, obs=pd.DataFrame({"label": y}))
    a.var["gene_name"] = syms
    a.var_names = syms
    print(f"[{task}] genes {len(syms)}/{len(gcols)}; N={len(df)}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    a = embed_data(a, str(CKPT), gene_col="gene_name", device=dev,
                   use_fast_transformer=False, return_new_adata=False)
    emb = np.asarray(a.obsm["X_scGPT"])

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y)
    clf = LogisticRegression(max_iter=2000).fit(emb[tr], y[tr])
    yp = clf.predict(emb[te])
    res = {"accuracy": float(accuracy_score(y[te], yp)),
           "macro_f1": float(f1_score(y[te], yp, average="macro")),
           "weighted_f1": float(f1_score(y[te], yp, average="weighted")),
           "n_test": len(te), "n_genes_used": len(syms)}
    print(f"[{task}] acc={res['accuracy']*100:.1f} macroF1={res['macro_f1']*100:.1f}", flush=True)
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "scGPT.json"
    res = json.loads(out.read_text()) if out.exists() else {"method": "scGPT (pan-cancer)", "tasks": {}}
    for t in TASKS:
        try:
            res["tasks"][t] = run_task(t)
        except Exception as e:
            import traceback; traceback.print_exc(); res["tasks"][t] = {"error": str(e)}
        out.write_text(json.dumps(res, indent=1))
    print("[scgpt] done ->", out)


if __name__ == "__main__":
    main()
