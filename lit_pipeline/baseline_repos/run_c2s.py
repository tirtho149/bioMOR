#!/usr/bin/env python3
"""Cell2Sentence baseline: frozen C2S-Pythia-410M cell embedding (mean of last
hidden state over the cell-sentence prompt) + logistic-regression probe on
labels. SAME seed-42 split as SMART. Writes results_dl_baselines/Cell2Sentence.json.
Run inside the c2s env.

Documented approximations:
  * gene id: column 'SYMBOL|entrez' -> SYMBOL (C2S cell sentences are gene names
    ranked by expression);
  * values are log2(x+1) RSEM (bulk), used directly to rank genes per sample;
  * bulk-on-a-single-cell LLM is OOD, reported transparently in the paper caption.
"""
import sys, json, shutil
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path("/work/mech-ai-scratch/tirtho/RecusrsiveQFormer")
sys.path.insert(0, str(REPO / "lit_pipeline/baseline_repos/Cell2Sentence/src"))
OUT = REPO / "results_dl_baselines"
CKPT = REPO / "lit_pipeline/baseline_repos/Cell2Sentence/checkpoints/C2S-Pythia-410m-cell-type-prediction"
TASKS = ["pathologic_stage", "pathologic_T", "pathologic_N", "os_binary", "tumor_status"]
SEED = 42


def run_task(task, work):
    import anndata as ad
    import cell2sentence as cs
    from cell2sentence.tasks import embed_cells
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    df = pd.read_csv(REPO / "data/tcga/unified_bio5.csv")
    meta = {"sample","cancer_type","cancer_name","os_binary","pathologic_stage",
            "pathologic_T","pathologic_N","tumor_status"}
    gcols = [c for c in df.columns if c not in meta]
    df = df.dropna(subset=[task]).reset_index(drop=True)

    syms, keep, seen = [], [], set()
    for c in gcols:
        s = c.split("|", 1)[0]
        if s and s != "?" and s not in seen:
            seen.add(s); keep.append(c); syms.append(s)
    X = df[keep].to_numpy(np.float32)
    y = df[task].astype(str).values
    # the cell_type_prediction prompt template needs 'organism' (model input) and
    # 'cell_type' (response). Only the input prompt drives the embedding, so the
    # cell_type value is irrelevant here; set it to the task label for traceability.
    a = ad.AnnData(X=X, obs=pd.DataFrame({"label": y, "organism": "human", "cell_type": y}))
    a.var_names = syms
    a.obs_names = [f"cell{i}" for i in range(len(y))]
    print(f"[{task}] genes {len(syms)}/{len(gcols)}; N={len(df)}", flush=True)

    wd = Path(work);
    if wd.exists(): shutil.rmtree(wd)
    wd.mkdir(parents=True, exist_ok=True)
    arrow, vocab = cs.CSData.adata_to_arrow(a, random_state=SEED,
                                            label_col_names=["label", "organism", "cell_type"])
    csdata = cs.CSData.csdata_from_arrow(arrow, vocab, save_dir=str(wd), save_name="csdata")
    csmodel = cs.CSModel(model_name_or_path=str(CKPT), save_dir=str(wd), save_name="model")
    emb = np.asarray(embed_cells(csdata, csmodel, n_genes=200, inference_batch_size=16))

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
    out = OUT / "Cell2Sentence.json"
    res = json.loads(out.read_text()) if out.exists() else {"method": "Cell2Sentence (Pythia-410M)", "tasks": {}}
    for t in TASKS:
        try:
            res["tasks"][t] = run_task(t, REPO / "results_dl_baselines/_c2s_work" / t)
        except Exception as e:
            import traceback; traceback.print_exc(); res["tasks"][t] = {"error": str(e)}
        out.write_text(json.dumps(res, indent=1))
    print("[c2s] done ->", out)


if __name__ == "__main__":
    main()
