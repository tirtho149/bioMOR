"""PathCNN baseline adapted to the 3-modality, 5-class TCGA BRCA PAM50 task.

This is a NEW file mirroring baseline_PathCNN/pathway_cnn_binary.py. Changes:
  * 3 modalities (cnv, expression, mutation) instead of 2 -> pathway image has
    n_pc(=2) PCs x 3 modalities = 6 channels  -> shape (N, n_pathways, 6).
  * 5-class softmax output + categorical_crossentropy.
  * Uses the shared data contract (brca_pam50_data) and the exact folds from
    D.make_folds(y). PCA / scalers are fit on TRAIN ONLY then applied to
    val/test (avoids leakage; the original fit PCA on the full matrix).
  * Reports metrics via multiclass_metrics.fold_metrics / write_metrics_json.

CNN architecture, optimizer (AdamW lr=1e-4 wd=5e-4), epochs (min 50 / max 200,
patience 25), batch size 16 follow the original pathway_cnn_binary.py.

Run from the repo root in the `pnet` conda env:
    cd /lustre/hdd/LAS/weile-lab/howlader/GraphPath_baselines
    conda activate pnet
    python baseline_PathCNN/pathway_cnn_multiclass.py
"""
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

import brca_pam50_data as D
from multiclass_metrics import fold_metrics, write_metrics_json

warnings.filterwarnings('ignore')

REPO = "/lustre/hdd/LAS/weile-lab/howlader/GraphPath_baselines"
PATHWAY_FILE = ("/lustre/hdd/LAS/weile-lab/howlader/Graph_Transformer/"
                "data_tcga/brca/filtered_pathways.csv")
RESULTS_JSON = f"{REPO}/baseline_PathCNN/results/brca_pam50/PathCNN_metrics.json"

MODALITY_ORDER = ["cnv", "expression", "mutation"]  # 3 modalities
N_PC = 2          # principal components per pathway per modality (as original)
NUM_CLASSES = 5


def create_pathway_gene_mapping(pathway_genes_df, available_genes):
    """Same parsing as the original: columns Pathway_ID, Pathway_Name, Genes
    (comma-separated). Keep pathways with >=5 genes present in `available_genes`.
    """
    print("Creating pathway-gene mapping...")
    available = set(available_genes)
    pathway_mapping, pathway_names = {}, {}

    for _, row in pathway_genes_df.iterrows():
        pathway_id = row['Pathway_ID']
        pathway_name = row['Pathway_Name']
        genes = str(row['Genes']).split(',')
        valid_genes = [g.strip() for g in genes if g.strip() in available]
        if len(valid_genes) >= 5:
            pathway_mapping[pathway_id] = valid_genes
            pathway_names[pathway_id] = pathway_name

    print(f"Found {len(pathway_mapping)} pathways with sufficient genes")
    return pathway_mapping, pathway_names


def fit_pca_to_pathways(train_matrix, pathway_mapping, gene_names, n_components=N_PC):
    """Fit a StandardScaler + PCA per pathway on the TRAIN matrix only.

    Returns dict pathway_idx -> (gene_indices, scaler, pca). Mirrors the
    original's per-pathway StandardScaler + randomized PCA, but fit on train.
    """
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    fitted = {}
    pathway_list = list(pathway_mapping.keys())

    for p_idx, p_id in enumerate(pathway_list):
        gene_indices = [gene_to_idx[g] for g in pathway_mapping[p_id]
                        if g in gene_to_idx]
        if not gene_indices:
            continue
        data = np.nan_to_num(train_matrix[:, gene_indices],
                             nan=0.0, posinf=0.0, neginf=0.0)
        if np.std(data) < 1e-10:
            continue
        n_comp = min(n_components, len(gene_indices), train_matrix.shape[0] - 1)
        if n_comp <= 0:
            continue
        try:
            scaler = StandardScaler()
            scaled = np.nan_to_num(scaler.fit_transform(data),
                                   nan=0.0, posinf=0.0, neginf=0.0)
            pca = PCA(n_components=n_comp, svd_solver='randomized', random_state=42)
            pca.fit(scaled)
            fitted[p_idx] = (gene_indices, scaler, pca, n_comp)
        except Exception as e:
            print(f"WARNING: PCA fit failed for pathway {p_id}: {e}")
            continue
    return fitted, pathway_list


def transform_pathways(matrix, fitted, n_pathways, n_components=N_PC):
    """Apply fitted scaler+PCA to `matrix`. Returns (N, n_pathways, n_components)."""
    out = np.zeros((matrix.shape[0], n_pathways, n_components))
    for p_idx, (gene_indices, scaler, pca, n_comp) in fitted.items():
        data = np.nan_to_num(matrix[:, gene_indices],
                             nan=0.0, posinf=0.0, neginf=0.0)
        scaled = np.nan_to_num(scaler.transform(data),
                               nan=0.0, posinf=0.0, neginf=0.0)
        out[:, p_idx, :n_comp] = pca.transform(scaled)
    return out


def stack_modalities(per_mod_pca, n_pc=N_PC):
    """per_mod_pca: list of (N, n_pathways, n_pc) arrays (one per modality).
    Concatenate along feature dim -> (N, n_pathways, n_pc * n_modalities).
    Mirrors the original create_pathway_images concatenation, extended to 3 mods.
    """
    return np.concatenate([m[:, :, :n_pc] for m in per_mod_pca], axis=2)


def create_pathcnn_model(input_shape, num_classes=NUM_CLASSES):
    """Original PathCNN architecture; only the final Dense width changes to 5."""
    image_input = Input(shape=input_shape)
    other_data_input = Input(shape=(1,))  # unused clinical placeholder (kept)

    conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu',
                   padding='same')(image_input)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    conv2 = MaxPooling2D(pool_size=(4, 2))(conv2)
    conv2 = Dropout(0.25)(conv2)
    flat = Flatten()(conv2)

    merged = Dense(64, activation='relu')(flat)
    merged = Dropout(0.5)(merged)
    predictions = Dense(num_classes, activation='softmax')(merged)

    return Model(inputs=[image_input, other_data_input], outputs=predictions)


def main():
    print("Starting PathCNN multiclass pipeline...")
    mods, y, patients, genes = D.load_modalities()
    print(f"Loaded {len(patients)} patients, {len(genes)} genes, "
          f"labels dist={np.bincount(y).tolist()}")

    pathway_df = pd.read_csv(PATHWAY_FILE)
    pathway_mapping, _ = create_pathway_gene_mapping(pathway_df, genes)
    n_pathways = len(pathway_mapping)

    folds = D.make_folds(y)

    # Training hyperparameters (from original).
    batch_size = 16
    min_epochs = 50
    max_epochs = 200
    patience = 25

    per_fold = []
    for fold_i, (tr_idx, val_idx, test_idx) in enumerate(folds, 1):
        print(f"\n{'='*60}\nFOLD {fold_i}/{len(folds)}\n{'='*60}")

        # Build pathway images per modality; fit PCA on TRAIN rows only.
        tr_mods, val_mods, te_mods = [], [], []
        for name in MODALITY_ORDER:
            M = mods[name]
            fitted, _ = fit_pca_to_pathways(M[tr_idx], pathway_mapping, genes)
            tr_mods.append(transform_pathways(M[tr_idx], fitted, n_pathways))
            val_mods.append(transform_pathways(M[val_idx], fitted, n_pathways))
            te_mods.append(transform_pathways(M[test_idx], fitted, n_pathways))

        X_train = stack_modalities(tr_mods)   # (Ntr, n_pathways, 6)
        X_val = stack_modalities(val_mods)
        X_test = stack_modalities(te_mods)
        print(f"Pathway image shape (train): {X_train.shape}")

        img_rows, img_cols = X_train.shape[1], X_train.shape[2]
        input_shape = (img_rows, img_cols, 1)
        X_train = X_train.reshape(-1, img_rows, img_cols, 1).astype('float32')
        X_val = X_val.reshape(-1, img_rows, img_cols, 1).astype('float32')
        X_test = X_test.reshape(-1, img_rows, img_cols, 1).astype('float32')

        y_train, y_val, y_test = y[tr_idx], y[val_idx], y[test_idx]
        y_train_cat = keras.utils.to_categorical(y_train, NUM_CLASSES)
        y_val_cat = keras.utils.to_categorical(y_val, NUM_CLASSES)

        dummy_train = np.zeros((len(X_train), 1))
        dummy_val = np.zeros((len(X_val), 1))
        dummy_test = np.zeros((len(X_test), 1))

        model = create_pathcnn_model(input_shape, NUM_CLASSES)
        try:
            from keras.optimizers import AdamW
            optimizer = AdamW(learning_rate=0.0001, weight_decay=5e-4,
                              beta_1=0.9, beta_2=0.999)
        except ImportError:
            optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Class weights for imbalance (multiclass): inverse frequency.
        counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(float)
        counts[counts == 0] = 1.0
        cw = counts.sum() / (NUM_CLASSES * counts)
        class_weight = {c: float(cw[c]) for c in range(NUM_CLASSES)}

        early_stopping = EarlyStopping(monitor='val_loss', patience=patience,
                                       restore_best_weights=True, verbose=1)

        history = model.fit([X_train, dummy_train], y_train_cat,
                            batch_size=batch_size, epochs=max_epochs, verbose=2,
                            class_weight=class_weight,
                            validation_data=([X_val, dummy_val], y_val_cat),
                            callbacks=[early_stopping])

        epochs_trained = len(history.history['loss'])
        if epochs_trained < min_epochs:
            model.fit([X_train, dummy_train], y_train_cat, batch_size=batch_size,
                      epochs=min_epochs - epochs_trained, verbose=2,
                      class_weight=class_weight,
                      validation_data=([X_val, dummy_val], y_val_cat))

        y_prob = model.predict([X_test, dummy_test], verbose=0)  # (Nte, 5)
        m = fold_metrics(y_test, y_prob, NUM_CLASSES)
        per_fold.append(m)
        print(f"Fold {fold_i} metrics: acc={m['accuracy']:.4f} "
              f"auc={m['auc']:.4f} f1={m['f1']:.4f} f1_macro={m['f1_macro']:.4f}")

    write_metrics_json(RESULTS_JSON, "PathCNN", "brca_pam50", NUM_CLASSES, per_fold)
    print(f"\nWrote metrics to {RESULTS_JSON}")
    return per_fold


if __name__ == "__main__":
    main()
