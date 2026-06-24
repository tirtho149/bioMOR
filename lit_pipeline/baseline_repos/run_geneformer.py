#!/usr/bin/env python3
"""Geneformer cell/disease-classification baseline on the genoNet bulk tasks.

Runs INSIDE the isolated geneformer env (transformers 4.x). Uses the SAME
seed-42 stratified 80/20 split as baselines.py / dl_baselines.py so the macro-F1
is apple-to-apple with SMART. Writes results_dl_baselines/Geneformer.json.

Adaptation of bulk TCGA -> a single-cell model (documented approximations):
  * gene id: column 'SYMBOL|entrez' -> Ensembl via Geneformer's gene_name_id_dict
    (symbol), NCBI gene2ensembl (entrez) as fallback  (~80% of genes covered);
  * counts: values are log2(x+1) RSEM, inverted to pseudo-counts 2^v - 1 for
    Geneformer's rank-value tokenizer;
  * modality: bulk samples treated as "cells" -- out-of-distribution for a
    single-cell model, reported transparently in the paper caption.

Usage (in geneformer env):
  python run_geneformer.py --task pathologic_stage --model <ckpt_dir> --tasks-all
"""
import argparse, json, pickle, os, shutil
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path("/work/mech-ai-scratch/tirtho/RecusrsiveQFormer")
GF = REPO / "lit_pipeline/baseline_repos/Geneformer"
AUX = REPO / "data/geneformer_aux"
OUT = REPO / "results_dl_baselines"
TASKS = ["pathologic_stage", "pathologic_T", "pathologic_N", "os_binary", "tumor_status"]
SEED = 42


def gene_mapping(gene_cols):
    name2id = pickle.load(open(GF / "geneformer/gene_name_id_dict_gc104M.pkl", "rb"))
    tokd = pickle.load(open(GF / "geneformer/token_dictionary_gc104M.pkl", "rb"))
    e2ens = json.load(open(AUX / "entrez2ensembl_human.json"))
    cols, ens = [], []
    for c in gene_cols:
        sym, ent = (c.split("|", 1) + [None])[:2] if "|" in c else (c, None)
        g = name2id.get(sym) if sym and sym != "?" else None
        if g is None and ent:
            g = e2ens.get(ent)
        if g and g in tokd:
            cols.append(c); ens.append(g)
    return cols, ens


def build_anndata(df, gene_cols, ens_ids, task, split_mask, path):
    import anndata as ad
    sub = df[gene_cols].to_numpy(dtype=np.float32)
    counts = np.expm1(sub * np.log(2.0))          # 2^v - 1  (invert log2(x+1) RSEM)
    counts = np.clip(np.rint(counts), 0, None).astype(np.float32)
    obs = pd.DataFrame({
        "label": df[task].astype(str).values,
        "split": np.where(split_mask, "train", "test"),
        "n_counts": counts.sum(1),
    })
    var = pd.DataFrame(index=np.arange(len(ens_ids)))
    var["ensembl_id"] = ens_ids
    a = ad.AnnData(X=counts, obs=obs, var=var)
    a = a[a.X.sum(1) > 0].copy()
    a.write_h5ad(path)
    return a


class _Collator:
    """Pad Geneformer input_ids to the batch max and build attention masks."""
    def __init__(self, pad_id):
        self.pad_id = pad_id
    def __call__(self, feats):
        import torch
        L = max(len(f["input_ids"]) for f in feats)
        ids, att, lab = [], [], []
        for f in feats:
            x = list(f["input_ids"])[:4096]
            pad = L - len(x)
            ids.append(x + [self.pad_id] * pad)
            att.append([1] * len(x) + [0] * pad)
            lab.append(int(f["labels"]))
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(att),
                "labels": torch.tensor(lab)}


def run_task(task, model_dir, work):
    """Fine-tune the Geneformer BERT directly with a HF Trainer (robust; this is
    what geneformer.Classifier does internally) on the seed-42 train split,
    evaluate on the held-out test split."""
    import sys; sys.path.insert(0, str(GF))
    from geneformer import TranscriptomeTokenizer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    import datasets, torch
    from transformers import (BertForSequenceClassification, Trainer, TrainingArguments)

    df = pd.read_csv(REPO / "data/tcga/unified_bio5.csv")
    meta = {"sample","cancer_type","cancer_name","os_binary","pathologic_stage",
            "pathologic_T","pathologic_N","tumor_status"}
    gene_cols_all = [c for c in df.columns if c not in meta]
    df = df.dropna(subset=[task]).reset_index(drop=True)
    cols, ens = gene_mapping(gene_cols_all)
    print(f"[{task}] usable genes {len(cols)}/{len(gene_cols_all)}; N={len(df)}", flush=True)

    y = df[task].astype(str).values
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y)
    mask = np.zeros(len(y), bool); mask[tr] = True

    wd = Path(work); wd.mkdir(parents=True, exist_ok=True)
    h5dir = wd / "h5"; h5dir.mkdir(exist_ok=True)
    build_anndata(df, cols, ens, task, mask, h5dir / f"{task}.h5ad")

    tok = TranscriptomeTokenizer({"label": "label", "split": "split"}, nproc=8)
    tok.tokenize_data(str(h5dir), str(wd), f"{task}", file_format="h5ad")
    ds = datasets.load_from_disk(str(wd / f"{task}.dataset"))

    classes = sorted(set(ds["label"]))
    c2i = {c: i for i, c in enumerate(classes)}
    i2c = {i: c for c, i in c2i.items()}
    ds = ds.map(lambda e: {"labels": c2i[e["label"]]})
    train_ds = ds.filter(lambda e: e["split"] == "train")
    test_ds = ds.filter(lambda e: e["split"] == "test")

    pad_id = pickle.load(open(GF / "geneformer/token_dictionary_gc104M.pkl", "rb")).get("<pad>", 0)
    model = BertForSequenceClassification.from_pretrained(str(model_dir), num_labels=len(classes))
    args = TrainingArguments(
        output_dir=str(wd / "trainer"), per_device_train_batch_size=8,
        per_device_eval_batch_size=16, num_train_epochs=8, learning_rate=5e-5,
        warmup_ratio=0.1, weight_decay=0.01, logging_steps=25,
        fp16=torch.cuda.is_available(), report_to=[], save_strategy="no", seed=SEED)
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      data_collator=_Collator(pad_id))
    trainer.train()

    pred = trainer.predict(test_ds)
    yp = [i2c[i] for i in pred.predictions.argmax(-1)]
    gt = [i2c[int(l)] for l in pred.label_ids]
    res = {"accuracy": float(accuracy_score(gt, yp)),
           "macro_f1": float(f1_score(gt, yp, average="macro")),
           "weighted_f1": float(f1_score(gt, yp, average="weighted")),
           "n_test": len(gt), "n_genes_used": len(cols), "n_classes": len(classes)}
    print(f"[{task}] acc={res['accuracy']*100:.1f} macroF1={res['macro_f1']*100:.1f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(GF / "Geneformer-V2-104M_CLcancer"))
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--work", default=str(REPO / "results_dl_baselines/_geneformer_work"))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "Geneformer.json"
    res = json.loads(out.read_text()) if out.exists() else {"method": "Geneformer (V2-104M_CLcancer)", "tasks": {}}
    for t in args.tasks:
        try:
            res["tasks"][t] = run_task(t, args.model, Path(args.work) / t)
        except Exception as e:
            import traceback; traceback.print_exc()
            res["tasks"][t] = {"error": str(e)}
        out.write_text(json.dumps(res, indent=1))
    print("[geneformer] done ->", out)


if __name__ == "__main__":
    main()
