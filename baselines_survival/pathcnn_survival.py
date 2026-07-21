"""PathCNN for 2-modality pan-cancer Cox survival prediction (Keras/TF).

Survival revision of the PathCNN baseline. Builds per-pathway PCA "images"
(n_pathways x [n_pc * 2 modalities]) exactly as the classification version, but
the CNN emits a single log-relative-hazard trained with a custom Cox partial
log-likelihood loss (within-batch Breslow approximation). Metric: Harrell's
C-index. PCA/scalers fit on TRAIN ONLY. Shared survival contract + 5-fold splits.

Run in the `pnet` conda env (TF/keras), with this dir on PYTHONPATH:
    python pathcnn_survival.py     (SMOKE=1 for a fast 1-fold smoke test)
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pan_survival_data as D
from survival_metrics import harrell_c_index, fold_metrics, write_metrics_json, write_risk_scores

PATHWAY_FILE = os.path.join(D.DATA_DIR, "filtered_pathways.csv")
RESULTS_JSON = os.path.join(HERE, os.environ.get("SURV_RESULTS_DIR", "results"), "PathCNN_survival_metrics.json")
RISK_CSV = os.path.join(HERE, os.environ.get("SURV_RESULTS_DIR", "results"), "PathCNN_risk_scores.csv")
SMOKE = os.environ.get("SMOKE", "0") == "1"

MODALITY_ORDER = D.MODALITY_NAMES  # 2- or 3-modal depending on SURV_3MODAL
N_PC = 2


def cox_loss(y_true, y_pred):
    """Negative Cox partial log-likelihood within the batch.
    y_true: (B,2) [time, event]; y_pred: (B,1) risk."""
    time = y_true[:, 0]
    event = y_true[:, 1]
    risk = y_pred[:, 0]
    order = tf.argsort(time, direction="DESCENDING")
    r = tf.gather(risk, order)
    e = tf.gather(event, order)
    log_cum = tf.math.cumulative_logsumexp(r)
    loss = -(r - log_cum) * e
    return tf.reduce_sum(loss) / (tf.reduce_sum(e) + 1e-8)


def create_pathway_gene_mapping(pathway_df, available_genes):
    available = set(available_genes)
    mapping = {}
    for _, row in pathway_df.iterrows():
        genes = [g.strip() for g in str(row["Genes"]).split(",") if g.strip() in available]
        if len(genes) >= 5:
            mapping[row["Pathway_ID"]] = genes
    print(f"Found {len(mapping)} pathways with sufficient genes")
    return mapping


def fit_pca_to_pathways(train_matrix, mapping, gene_names, n_components=N_PC):
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    fitted = {}
    for p_idx, p_id in enumerate(mapping.keys()):
        gi = [gene_to_idx[g] for g in mapping[p_id] if g in gene_to_idx]
        if not gi:
            continue
        data = np.nan_to_num(train_matrix[:, gi])
        if np.std(data) < 1e-10:
            continue
        n_comp = min(n_components, len(gi), train_matrix.shape[0] - 1)
        if n_comp <= 0:
            continue
        scaler = StandardScaler()
        scaled = np.nan_to_num(scaler.fit_transform(data))
        pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=42)
        pca.fit(scaled)
        fitted[p_idx] = (gi, scaler, pca, n_comp)
    return fitted


def transform_pathways(matrix, fitted, n_pathways, n_components=N_PC):
    out = np.zeros((matrix.shape[0], n_pathways, n_components), dtype=np.float32)
    for p_idx, (gi, scaler, pca, n_comp) in fitted.items():
        scaled = np.nan_to_num(scaler.transform(np.nan_to_num(matrix[:, gi])))
        out[:, p_idx, :n_comp] = pca.transform(scaled)
    return out


def create_model(input_shape):
    image_input = Input(shape=input_shape)
    dummy_input = Input(shape=(1,))
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(image_input)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D(pool_size=(4, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.5)(x)
    risk = Dense(1, activation="linear", use_bias=False)(x)  # Cox PH head
    return Model(inputs=[image_input, dummy_input], outputs=risk)


def main():
    print("Starting PathCNN survival pipeline...")
    mods, time, event, patients, genes = D.load_modalities()
    print(f"patients={len(patients)} genes={len(genes)} events={int(event.sum())}")

    mapping = create_pathway_gene_mapping(pd.read_csv(PATHWAY_FILE), genes)
    n_pathways = len(mapping)

    folds = D.make_folds(event)
    if SMOKE:
        folds = folds[:1]

    batch_size = D.BATCH_SIZE
    min_epochs = D.MIN_EPOCHS
    max_epochs = D.MAX_EPOCHS
    patience = D.PATIENCE

    per_fold = []
    risk_records = []
    for fold_i, (tr_idx, val_idx, test_idx) in enumerate(folds, 1):
        print(f"\n{'='*60}\nFOLD {fold_i}/{len(folds)}\n{'='*60}")
        tr_mods, val_mods, te_mods = [], [], []
        for name in MODALITY_ORDER:
            M = mods[name]
            fitted = fit_pca_to_pathways(M[tr_idx], mapping, genes)
            tr_mods.append(transform_pathways(M[tr_idx], fitted, n_pathways))
            val_mods.append(transform_pathways(M[val_idx], fitted, n_pathways))
            te_mods.append(transform_pathways(M[test_idx], fitted, n_pathways))

        X_train = np.concatenate(tr_mods, axis=2)   # (Ntr, n_pathways, 2*N_PC)
        X_val = np.concatenate(val_mods, axis=2)
        X_test = np.concatenate(te_mods, axis=2)
        rows, cols = X_train.shape[1], X_train.shape[2]
        input_shape = (rows, cols, 1)
        X_train = X_train.reshape(-1, rows, cols, 1).astype("float32")
        X_val = X_val.reshape(-1, rows, cols, 1).astype("float32")
        X_test = X_test.reshape(-1, rows, cols, 1).astype("float32")
        print(f"Pathway image shape (train): {X_train.shape}")

        yt_train = np.stack([time[tr_idx], event[tr_idx]], axis=1).astype("float32")
        yt_val = np.stack([time[val_idx], event[val_idx]], axis=1).astype("float32")
        dummy_tr = np.zeros((len(X_train), 1))
        dummy_val = np.zeros((len(X_val), 1))
        dummy_te = np.zeros((len(X_test), 1))

        model = create_model(input_shape)
        from keras.optimizers import AdamW
        model.compile(optimizer=AdamW(learning_rate=D.LR, weight_decay=D.WEIGHT_DECAY), loss=cox_loss)

        es = EarlyStopping(monitor="val_loss", patience=patience,
                           restore_best_weights=True, verbose=1)
        hist = model.fit([X_train, dummy_tr], yt_train, batch_size=batch_size,
                         epochs=max_epochs, verbose=2, shuffle=True,
                         validation_data=([X_val, dummy_val], yt_val), callbacks=[es])
        if len(hist.history["loss"]) < min_epochs:
            model.fit([X_train, dummy_tr], yt_train, batch_size=batch_size,
                      epochs=min_epochs - len(hist.history["loss"]), verbose=2,
                      shuffle=True, validation_data=([X_val, dummy_val], yt_val))

        risk = model.predict([X_test, dummy_te], verbose=0).reshape(-1)
        m = fold_metrics(risk, time[test_idx], event[test_idx])
        per_fold.append(m)
        risk_records.append({"test_idx": test_idx, "risk": risk,
                             "time": time[test_idx], "event": event[test_idx]})
        print(f"Fold {fold_i} C-index={m['c_index']:.4f} (n={m['n']}, events={m['events']})")

    write_metrics_json(RESULTS_JSON, "PathCNN", os.environ.get("SURV_DATASET_TAG", "pan_survival_cox"), per_fold)
    write_risk_scores(RISK_CSV, patients, risk_records)
    print(f"\nWrote metrics to {RESULTS_JSON}")
    print(f"Wrote risk scores to {RISK_CSV}")


if __name__ == "__main__":
    main()
