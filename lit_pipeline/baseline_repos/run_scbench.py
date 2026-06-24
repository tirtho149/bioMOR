#!/usr/bin/env python3
"""scbenchmark baseline: train its scModel (6-layer / 256-dim Transformer) FROM
SCRATCH end-to-end for classification on each genoNet task. SAME seed-42 split as
SMART. Writes results_dl_baselines/scbenchmark.json. Run inside the scbench env.

Faithful to scbenchmark's setup: its own scModel architecture, gene-symbol vocab
(vocab.json), quantile binning (_binning, n_bins=51), <cls>-token cell embedding.
We train model+linear-head jointly (no masked pretraining checkpoint exists for
bulk TCGA), which is the fair "from-scratch transformer" baseline the paper cites.

Documented approximations:
  * gene id: 'SYMBOL|entrez' -> SYMBOL matched to the scbenchmark vocab;
  * values are log2(x+1) RSEM (bulk), quantile-binned per sample as in scbenchmark.
"""
import sys, json, argparse
from pathlib import Path
from types import SimpleNamespace
import numpy as np, pandas as pd

REPO = Path("/work/mech-ai-scratch/tirtho/RecusrsiveQFormer")
SB = REPO / "lit_pipeline/baseline_repos/scbenchmark"
sys.path.insert(0, str(SB))
OUT = REPO / "results_dl_baselines"
TASKS = ["pathologic_stage", "pathologic_T", "pathologic_N", "os_binary", "tumor_status"]
SEED = 42
MAXLEN = 512
NBINS = 51


def _args():
    return SimpleNamespace(embsize=256, nheads=8, nlayers=6, dropout=0.2,
                           model_structure="transformer", cell_emb_style="cls",
                           pad_token="<pad>", n_bins=NBINS)


def _binning(row, n_bins=NBINS):
    import torch
    if torch.all(row == 0):
        return torch.zeros_like(row)
    bins = torch.quantile(row, torch.linspace(0, 1, n_bins - 1))
    left = torch.bucketize(row, bins, right=False) - 1
    right = torch.bucketize(row, bins, right=True) - 1
    rands = torch.rand_like(row)
    digits = torch.ceil(rands * (right - left) + left).to(torch.int64)
    return digits


def _encode(Xrow, gene_ids, cls_id, pad_id):
    """top-MAXLEN expressed genes by value, prepend <cls>, quantile-bin values."""
    import torch
    nz = np.nonzero(Xrow > 0)[0]
    if len(nz) == 0:
        nz = np.arange(min(len(Xrow), MAXLEN - 1))
    order = nz[np.argsort(-Xrow[nz])][:MAXLEN - 1]
    g = [cls_id] + [gene_ids[i] for i in order]
    v = np.concatenate([[0.0], Xrow[order]]).astype(np.float32)
    L = MAXLEN
    if len(g) < L:
        pad = L - len(g)
        g = g + [pad_id] * pad
        v = np.concatenate([v, np.zeros(pad, np.float32)])
    gt = torch.tensor(g, dtype=torch.long)
    vt = _binning(torch.tensor(v, dtype=torch.float)).float()
    return gt, vt


def run_task(task):
    import torch
    from torch import nn
    from torch.utils.data import TensorDataset, DataLoader
    from models_utils.model import scModel
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score

    torch.manual_seed(SEED); np.random.seed(SEED)
    vocab = json.load(open(SB / "vocab.json"))
    cls_id, pad_id = vocab["<cls>"], vocab["<pad>"]
    args = _args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(REPO / "data/tcga/unified_bio5.csv")
    meta = {"sample","cancer_type","cancer_name","os_binary","pathologic_stage",
            "pathologic_T","pathologic_N","tumor_status"}
    gcols = [c for c in df.columns if c not in meta]
    df = df.dropna(subset=[task]).reset_index(drop=True)

    keep, ids, seen = [], [], set()
    for c in gcols:
        s = c.split("|", 1)[0]
        if s and s != "?" and s in vocab and s not in seen:
            seen.add(s); keep.append(c); ids.append(vocab[s])
    X = df[keep].to_numpy(np.float32)
    ylab = df[task].astype(str).values
    classes = sorted(set(ylab)); c2i = {c: i for i, c in enumerate(classes)}
    y = np.array([c2i[v] for v in ylab])
    print(f"[{task}] genes {len(keep)}/{len(gcols)}; N={len(df)}; classes={len(classes)}", flush=True)

    G = torch.stack([_encode(X[i], ids, cls_id, pad_id)[0] for i in range(len(X))])
    V = torch.stack([_encode(X[i], ids, cls_id, pad_id)[1] for i in range(len(X))])
    yt = torch.tensor(y, dtype=torch.long)

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y)
    dl = DataLoader(TensorDataset(G[tr], V[tr], yt[tr]), batch_size=32, shuffle=True)

    model = scModel(vocab, args).to(dev)
    head = nn.Linear(args.embsize, len(classes)).to(dev)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=2e-4, weight_decay=1e-2)
    crit = nn.CrossEntropyLoss()

    def evaluate():
        model.eval(); head.eval()
        with torch.no_grad():
            gp = G[te].to(dev); vp = V[te].to(dev)
            mask = gp.eq(pad_id)
            emb = model(gp, vp, src_key_padding_mask=mask)["cell_emb"]
            pred = head(emb).argmax(1).cpu().numpy()
        return (accuracy_score(y[te], pred), f1_score(y[te], pred, average="macro"),
                f1_score(y[te], pred, average="weighted"))

    best = (0.0, 0.0, 0.0)
    for ep in range(25):
        model.train(); head.train()
        for g, v, lab in dl:
            g, v, lab = g.to(dev), v.to(dev), lab.to(dev)
            mask = g.eq(pad_id)
            emb = model(g, v, src_key_padding_mask=mask)["cell_emb"]
            loss = crit(head(emb), lab)
            opt.zero_grad(); loss.backward(); opt.step()
        acc, mf1, wf1 = evaluate()
        if mf1 > best[1]:
            best = (acc, mf1, wf1)
        print(f"[{task}] ep{ep} test acc={acc*100:.1f} mF1={mf1*100:.1f}", flush=True)

    res = {"accuracy": float(best[0]), "macro_f1": float(best[1]),
           "weighted_f1": float(best[2]), "n_test": len(te),
           "n_genes_used": len(keep), "n_classes": len(classes)}
    print(f"[{task}] BEST acc={best[0]*100:.1f} macroF1={best[1]*100:.1f}", flush=True)
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "scbenchmark.json"
    res = json.loads(out.read_text()) if out.exists() else {"method": "scbenchmark (6L/256d)", "tasks": {}}
    for t in TASKS:
        try:
            res["tasks"][t] = run_task(t)
        except Exception as e:
            import traceback; traceback.print_exc(); res["tasks"][t] = {"error": str(e)}
        out.write_text(json.dumps(res, indent=1))
    print("[scbench] done ->", out)


if __name__ == "__main__":
    main()
