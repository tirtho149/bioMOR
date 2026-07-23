#!/usr/bin/env python3
"""scTransSort CV runner — bioMOR baseline adapter.

Thin adapter around upstream scTransSort (sc_baselines/scTransSort/model/trans_model,
a TensorFlow/Keras ViT). We reuse its `VisionTransformer` class verbatim (loaded from
the extension-less upstream file via importlib), and reproduce its gene->image
embedding from model/read: each cell's gene-expression vector is zero-padded and
reshaped into an L x L x 3 "image" (L=ceil(sqrt(G))), then classified by the ViT.

Upstream hardcodes img_size=224 in its vit_base_* helpers, which does not match our
gene counts. The `VisionTransformer` class itself is fully parametric on
(img_size, patch_size), so we instantiate it directly with img_size=L and a
patch_size that divides L — keeping the model code untouched, only choosing config.

Data/labels/CV come from biomor_common (shared X/y arrays + seed-42 CV5 folds);
train on train+val, predict test per fold; write common scores CSV via bc.write_scores.

Ref: Jiao et al., "scTransSort: Transformers for intelligent annotation of cell types
by gene embeddings" (Biomolecules 2023).
"""
from __future__ import annotations
import argparse, importlib.util, math, os, random, sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
SCT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import biomor_common as bc

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

# --- load upstream VisionTransformer from the extension-less file, unchanged ---
_TRANS_MODEL = SCT_DIR / "model" / "trans_model"
_spec = importlib.util.spec_from_loader(
    "sct_trans_model",
    importlib.machinery.SourceFileLoader("sct_trans_model", str(_TRANS_MODEL)))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
VisionTransformer = _mod.VisionTransformer


def pick_patch_size(L, prefer=(16, 14, 12, 10, 8, 7, 6, 5, 4, 3, 2, 1)):
    """Grow L to the next value divisible by a reasonable patch size."""
    for p in prefer:
        Lp = int(math.ceil(L / p) * p)
        if Lp // p >= 4:            # need >=16 patches for a meaningful ViT
            return Lp, p
    return L, 1


def to_images(X, L):
    """model/read changefeature: pad each gene vector to L*L, reshape to (L,L),
    replicate to 3 channels -> (N, L, L, 3)."""
    N, G = X.shape
    pad = L * L - G
    if pad > 0:
        X = np.pad(X, ((0, 0), (0, pad)), mode="constant")
    elif pad < 0:
        X = X[:, : L * L]
    imgs = X.reshape(N, L, L, 1).astype(np.float32)
    imgs = np.repeat(imgs, 3, axis=3)
    return imgs


def run_fold(X, y, tr, va, te, embed_dim, depth, num_heads, epochs, batch_size,
             lr, weight_decay, seed=66):
    np.random.seed(seed); random.seed(seed); tf.random.set_seed(seed)

    tr_all = np.concatenate([tr, va])
    Xtr, ytr = X[tr_all], y[tr_all].astype(np.int64)
    Xte, yte = X[te], y[te].astype(np.int64)

    G = X.shape[1]
    L0 = int(math.sqrt(G)) + 1
    L, patch = pick_patch_size(L0)
    num_classes = int(y.max()) + 1
    print(f"    G={G} img={L}x{L} patch={patch} classes={num_classes}", flush=True)

    Itr = to_images(Xtr, L)
    Ite = to_images(Xte, L)

    model = VisionTransformer(img_size=L, patch_size=patch, embed_dim=embed_dim,
                              depth=depth, num_heads=num_heads,
                              representation_size=None, num_classes=num_classes,
                              name="scTransSort")
    model.build((batch_size, L, L, 3))

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)

    tr_ds = (tf.data.Dataset.from_tensor_slices((Itr, ytr))
             .shuffle(len(Itr), seed=seed)
             .batch(batch_size, drop_remainder=True))

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            out = model(images, training=True)
            ce = loss_object(labels, out)
            l2 = weight_decay * tf.add_n(
                [tf.nn.l2_loss(v) for v in model.trainable_variables
                 if "bias" not in v.name and "gamma" not in v.name and "beta" not in v.name])
            loss = ce + l2
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return ce

    for ep in range(epochs):
        tot = 0.0; nb = 0
        lr_e = ((1 + math.cos(ep * math.pi / epochs)) / 2) * (1 - 0.01) + 0.01
        opt.learning_rate = lr_e * lr
        for images, labels in tr_ds:
            tot += float(train_step(images, labels)); nb += 1
        print(f"    epoch {ep+1}/{epochs} loss={tot/max(nb,1):.4f}", flush=True)

    # predict test
    preds = []
    te_ds = tf.data.Dataset.from_tensor_slices(Ite).batch(batch_size)
    for images in te_ds:
        out = model(images, training=False)
        preds.append(tf.argmax(out, axis=1).numpy())
    yp = np.concatenate(preds)
    f1, acc = bc.fold_metrics(yte, yp)
    return f1, acc, len(te)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--embed_dim", type=int, default=64)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--work_dir", default=None)
    args = ap.parse_args()

    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try: tf.config.experimental.set_memory_growth(g, True)
        except Exception: pass
    print(f"[scTransSort] dataset={args.dataset} GPUs={len(gpus)} "
          f"embed_dim={args.embed_dim} depth={args.depth} epochs={args.epochs}",
          flush=True)

    X, y, genes = bc.load_sc(args.dataset)
    print(f"  loaded X{X.shape} classes={int(y.max())+1}", flush=True)
    folds = bc.load_sc_folds(args.dataset, y)

    f1s, accs, nts = [], [], []
    for i, (tr, va, te) in enumerate(folds[:args.folds]):
        print(f"  fold {i+1}/{min(args.folds,len(folds))}", flush=True)
        f1, acc, nt = run_fold(X, y, tr, va, te, args.embed_dim, args.depth,
                               args.num_heads, args.epochs, args.batch_size,
                               args.lr, args.weight_decay)
        print(f"  fold {i+1} macro_f1={f1:.2f} acc={acc:.2f}", flush=True)
        f1s.append(f1); accs.append(acc); nts.append(nt)

    wd = args.work_dir or str(SCT_DIR / "work_dirs" / args.dataset)
    out = bc.write_scores(wd, "scTransSort", args.dataset, f1s, accs, nts)
    print(f"[scTransSort] wrote {out}", flush=True)
    print(f"[scTransSort] mean macro_f1={np.mean(f1s):.2f} acc={np.mean(accs):.2f}", flush=True)


if __name__ == "__main__":
    main()
