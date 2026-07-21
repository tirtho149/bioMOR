import json
import os
import warnings

import keras
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    pairwise_distances,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

PLOT_FIGSIZE = (3.54, 3.54)
PLOT_DPI = 600
PLOT_FONT_SIZE = 12
matplotlib.rcParams.update(
    {
        "font.size": PLOT_FONT_SIZE,
        "axes.titlesize": PLOT_FONT_SIZE,
        "axes.labelsize": PLOT_FONT_SIZE,
        "xtick.labelsize": PLOT_FONT_SIZE,
        "ytick.labelsize": PLOT_FONT_SIZE,
        "legend.fontsize": PLOT_FONT_SIZE,
    }
)


def load_and_preprocess_data():
    """
    Load and preprocess mutation, CNV, labels, and pathway data.
    Ensure mutation and CNV data have the same genes in the same order.
    """
    print("Loading data...")

    mutation_data = pd.read_csv("mutation_data.csv")
    cnv_data = pd.read_csv("cnv_data.csv")
    labels_data = pd.read_csv("labels.csv")
    pathway_genes = pd.read_csv("pathway_genes.csv")

    mutation_patients = mutation_data.iloc[:, 0].values
    cnv_patients = cnv_data.iloc[:, 0].values
    label_patients = labels_data.iloc[:, 0].values

    common_patients = list(set(mutation_patients) & set(cnv_patients) & set(label_patients))
    print(f"Found {len(common_patients)} common patients across all datasets")

    mutation_aligned = mutation_data[mutation_data.iloc[:, 0].isin(common_patients)].reset_index(drop=True)
    cnv_aligned = cnv_data[cnv_data.iloc[:, 0].isin(common_patients)].reset_index(drop=True)
    labels_aligned = labels_data[labels_data.iloc[:, 0].isin(common_patients)].reset_index(drop=True)

    mutation_aligned = mutation_aligned.sort_values(by=mutation_aligned.columns[0]).reset_index(drop=True)
    cnv_aligned = cnv_aligned.sort_values(by=cnv_aligned.columns[0]).reset_index(drop=True)
    labels_aligned = labels_aligned.sort_values(by=labels_aligned.columns[0]).reset_index(drop=True)

    mutation_genes = set(mutation_aligned.columns[1:].tolist())
    cnv_genes = set(cnv_aligned.columns[1:].tolist())

    common_genes = sorted(list(mutation_genes & cnv_genes))
    print(f"Found {len(common_genes)} common genes between mutation and CNV data")

    mutation_aligned = (
        mutation_aligned[["patient_id"] + common_genes]
        if "patient_id" in mutation_aligned.columns
        else mutation_aligned[[mutation_aligned.columns[0]] + common_genes]
    )
    cnv_aligned = (
        cnv_aligned[["patient_id"] + common_genes]
        if "patient_id" in cnv_aligned.columns
        else cnv_aligned[[cnv_aligned.columns[0]] + common_genes]
    )

    gene_names = common_genes
    mutation_matrix = mutation_aligned.iloc[:, 1:].values
    cnv_matrix = cnv_aligned.iloc[:, 1:].values
    labels = labels_aligned.iloc[:, 1].values.astype(int)

    print(f"Mutation data shape: {mutation_matrix.shape}")
    print(f"CNV data shape: {cnv_matrix.shape}")
    print(f"Number of common genes: {len(gene_names)}")
    print(f"Label distribution: {np.bincount(labels)}")

    assert mutation_matrix.shape[1] == cnv_matrix.shape[1], "Mutation and CNV matrices must have same number of genes"
    assert len(gene_names) == mutation_matrix.shape[1], "Gene names length must match data matrix columns"

    return mutation_matrix, cnv_matrix, labels, gene_names, pathway_genes


def create_pathway_gene_mapping(pathway_genes_df, available_genes):
    """Create pathway to gene mapping and keep pathways with >= 5 genes."""
    print("Creating pathway-gene mapping...")

    pathway_mapping = {}
    pathway_names = {}

    for _, row in pathway_genes_df.iterrows():
        pathway_id = row["Pathway_ID"]
        pathway_name = row["Pathway_Name"]
        genes = str(row["Genes"]).split(",")

        valid_genes = [gene.strip() for gene in genes if gene.strip() in available_genes]

        if len(valid_genes) >= 5:
            pathway_mapping[pathway_id] = valid_genes
            pathway_names[pathway_id] = pathway_name

    print(f"Found {len(pathway_mapping)} pathways with sufficient genes")
    return pathway_mapping, pathway_names


def apply_pca_to_pathways(data_matrix, pathway_mapping, gene_names, n_components=5):
    """Apply PCA to each pathway separately with robust checks."""
    print(f"Applying PCA to pathways... (data shape: {data_matrix.shape})")

    n_samples = data_matrix.shape[0]
    n_pathways = len(pathway_mapping)
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}

    if np.isnan(data_matrix).any() or np.isinf(data_matrix).any():
        data_matrix = np.nan_to_num(data_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    pathway_pca_data = np.zeros((n_samples, n_pathways, n_components))
    pathway_list = list(pathway_mapping.keys())

    failed_pathways = []

    for pathway_idx, pathway_id in enumerate(pathway_list):
        pathway_genes = pathway_mapping[pathway_id]
        gene_indices = [gene_to_idx[gene] for gene in pathway_genes if gene in gene_to_idx]

        if len(gene_indices) == 0:
            continue

        if max(gene_indices) >= data_matrix.shape[1]:
            continue

        pathway_data = data_matrix[:, gene_indices]
        pathway_data = np.nan_to_num(pathway_data, nan=0.0, posinf=0.0, neginf=0.0)

        if np.std(pathway_data) < 1e-10:
            continue

        n_comp = min(n_components, len(gene_indices), n_samples - 1)
        if n_comp <= 0:
            continue

        try:
            scaler = StandardScaler()
            pathway_data_scaled = scaler.fit_transform(pathway_data)
            pathway_data_scaled = np.nan_to_num(pathway_data_scaled, nan=0.0, posinf=0.0, neginf=0.0)

            pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=42)
            pca_result = pca.fit_transform(pathway_data_scaled)
            pathway_pca_data[:, pathway_idx, :n_comp] = pca_result
        except Exception:
            failed_pathways.append(pathway_id)

    if failed_pathways:
        msg = failed_pathways[:5]
        print(f"PCA failed for {len(failed_pathways)} pathways. Examples: {msg}")

    print(f"PCA completed. Result shape: {pathway_pca_data.shape}")
    return pathway_pca_data, pathway_list


def create_pathway_images(mutation_pca, cnv_pca, n_pc=2):
    """Create pathway images by combining mutation and CNV PCs."""
    print(f"Creating pathway images with {n_pc} principal components...")

    n_samples, n_pathways, _ = mutation_pca.shape
    pathway_images = np.zeros((n_samples, n_pathways, n_pc * 2))

    for i in range(n_samples):
        mutation_pc = mutation_pca[i, :, :n_pc]
        cnv_pc = cnv_pca[i, :, :n_pc]
        pathway_images[i, :, :] = np.concatenate([mutation_pc, cnv_pc], axis=1)

    return pathway_images


def order_pathways_by_correlation(pathway_images):
    """Order pathways by correlation so similar pathways are nearby."""
    print("Ordering pathways by correlation...")

    n_samples, n_pathways, n_features = pathway_images.shape
    flattened_data = pathway_images.reshape(n_pathways, n_samples * n_features)
    correlation_matrix = np.corrcoef(flattened_data)

    ordered_indices = [0]
    remaining_indices = list(range(1, n_pathways))

    while remaining_indices:
        last_pathway = ordered_indices[-1]
        correlations = [correlation_matrix[last_pathway, idx] for idx in remaining_indices]
        next_pathway_pos = np.argmax(correlations)
        next_pathway = remaining_indices.pop(next_pathway_pos)
        ordered_indices.append(next_pathway)

    reordered_images = pathway_images[:, ordered_indices, :]
    return reordered_images, ordered_indices


def create_pathcnn_model(input_shape, num_classes=2):
    """Create PathCNN model."""
    image_input = Input(shape=input_shape, name="image_input")
    other_data_input = Input(shape=(1,), name="other_data_input")

    conv1 = Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same")(image_input)
    conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv1)
    conv2 = MaxPooling2D(pool_size=(4, 2))(conv2)
    conv2 = Dropout(0.25)(conv2)
    first_part_output = Flatten()(conv2)

    merged_model = Dense(64, activation="relu", name="embedding_dense")(first_part_output)
    merged_model = Dropout(0.5, name="embedding_dropout")(merged_model)
    predictions = Dense(num_classes, activation="softmax", name="classifier")(merged_model)

    model = Model(inputs=[image_input, other_data_input], outputs=predictions)
    return model


def calculate_metrics(y_true, y_pred_proba, y_pred_binary, pos_label=1):
    """Calculate binary classification metrics safely."""
    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred_binary)

    try:
        metrics["auc"] = roc_auc_score(y_true, y_pred_proba)
    except Exception:
        metrics["auc"] = np.nan

    try:
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        metrics["aupr"] = auc(recall, precision)
    except Exception:
        metrics["aupr"] = np.nan

    metrics["f1"] = f1_score(y_true, y_pred_binary, average="binary", pos_label=pos_label, zero_division=0)
    metrics["precision"] = precision_score(
        y_true, y_pred_binary, average="binary", pos_label=pos_label, zero_division=0
    )
    metrics["recall"] = recall_score(
        y_true, y_pred_binary, average="binary", pos_label=pos_label, zero_division=0
    )
    return metrics


def _safe_2d_projection(embeddings, method="umap", random_state=42):
    """Project embeddings to 2D using UMAP (fallback TSNE -> PCA)."""
    embeddings = np.asarray(embeddings)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(-1, 1)

    n_samples, n_features = embeddings.shape
    if n_samples < 2:
        return np.zeros((n_samples, 2), dtype=np.float32), "degenerate"

    pca_dim = min(50, n_features, n_samples - 1)
    if pca_dim < 1:
        return np.zeros((n_samples, 2), dtype=np.float32), "degenerate"
    emb_in = PCA(n_components=pca_dim, random_state=random_state).fit_transform(embeddings)

    method = method.lower()
    if method == "umap":
        try:
            import umap  # type: ignore

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=max(2, min(15, n_samples - 1)),
                min_dist=0.15,
                metric="euclidean",
                random_state=random_state,
            )
            return reducer.fit_transform(emb_in), "umap"
        except Exception:
            method = "tsne"

    if method == "tsne":
        perplexity = max(2, min(30, n_samples - 1))
        proj = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=random_state,
            init="pca",
            learning_rate="auto",
        ).fit_transform(emb_in)
        return proj, "tsne"

    return PCA(n_components=2, random_state=random_state).fit_transform(embeddings), "pca"


def analyze_embedding_separation(embeddings, y_true, sample_ids, outdir, split_name, random_state=42):
    """Create projection and distance analysis plots for one split."""
    if embeddings.size == 0 or len(y_true) == 0:
        print(f"No embeddings available for split '{split_name}'")
        return None

    split_dir = os.path.join(outdir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    proj_2d, method_used = _safe_2d_projection(embeddings, method="umap", random_state=random_state)
    class_names = np.array(["Class 0" if int(v) == 0 else "Class 1" for v in y_true])

    proj_df = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "x": proj_2d[:, 0],
            "y": proj_2d[:, 1],
            "class_label": class_names,
            "y_true": y_true,
        }
    )

    proj_csv = os.path.join(split_dir, f"{split_name}_embedding_projection.csv")
    proj_df.to_csv(proj_csv, index=False)

    plt.figure(figsize=PLOT_FIGSIZE)
    ax = sns.scatterplot(data=proj_df, x="x", y="y", hue="class_label", palette="Set1", alpha=0.9, s=10)
    plt.title(f"{split_name.capitalize()} Embedding Projection by Ground Truth ({method_used.upper()})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    ax.legend(
        title="",
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        markerscale=0.7,
        handletextpad=0.3,
        columnspacing=0.6,
        borderaxespad=0.0,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    proj_plot = os.path.join(split_dir, f"{split_name}_projection_{method_used}.svg")
    plt.savefig(proj_plot, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()

    dist_matrix = pairwise_distances(embeddings, metric="euclidean")
    order = np.argsort(y_true)
    sorted_dist = dist_matrix[order][:, order]

    heatmap_path = os.path.join(split_dir, f"{split_name}_distance_heatmap.svg")
    plt.figure(figsize=PLOT_FIGSIZE)
    sns.heatmap(sorted_dist, cmap="viridis", cbar_kws={"label": "Euclidean distance"})
    plt.title(f"{split_name.capitalize()}: Pairwise Distance Heatmap (sorted by ground truth)")
    plt.xlabel(f"{split_name.capitalize()} samples (sorted)")
    plt.ylabel(f"{split_name.capitalize()} samples (sorted)")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()

    triu_i, triu_j = np.triu_indices(len(y_true), k=1)
    pair_d = dist_matrix[triu_i, triu_j]
    same_mask = y_true[triu_i] == y_true[triu_j]
    within = pair_d[same_mask]
    between = pair_d[~same_mask]

    dist_rows = []
    if within.size > 0:
        dist_rows.extend([{"type": "within_class", "distance": float(v)} for v in within])
    if between.size > 0:
        dist_rows.extend([{"type": "between_class", "distance": float(v)} for v in between])
    dist_df = pd.DataFrame(dist_rows)

    dist_csv = os.path.join(split_dir, f"{split_name}_pairwise_distance_distribution.csv")
    dist_df.to_csv(dist_csv, index=False)

    dist_plot = os.path.join(split_dir, f"{split_name}_distance_distribution.svg")
    if not dist_df.empty:
        plt.figure(figsize=PLOT_FIGSIZE)
        sns.boxplot(data=dist_df, x="type", y="distance", palette="Set2")
        sns.stripplot(data=dist_df, x="type", y="distance", color="black", alpha=0.25, size=2)
        plt.title(f"{split_name.capitalize()}: Within vs Between Class Distances")
        plt.xlabel("")
        plt.ylabel("Euclidean distance")
        plt.tight_layout()
        plt.savefig(dist_plot, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()

    summary = {
        "split": split_name,
        "projection_method": method_used,
        "n_samples": int(len(y_true)),
        "mean_within_class_distance": float(np.mean(within)) if within.size > 0 else None,
        "std_within_class_distance": float(np.std(within)) if within.size > 0 else None,
        "mean_between_class_distance": float(np.mean(between)) if between.size > 0 else None,
        "std_between_class_distance": float(np.std(between)) if between.size > 0 else None,
        "between_over_within_ratio": (
            float(np.mean(between) / (np.mean(within) + 1e-12)) if (within.size > 0 and between.size > 0) else None
        ),
        "projection_csv": proj_csv,
        "projection_plot": proj_plot,
        "distance_heatmap": heatmap_path,
        "distance_distribution_csv": dist_csv,
        "distance_distribution_plot": dist_plot,
    }

    summary_path = os.path.join(split_dir, f"{split_name}_embedding_separation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"{split_name.capitalize()} embedding-separation analysis saved to: {split_dir}")
    return summary


def train_pathcnn_single_split(pathway_images, labels, random_state=42):
    """Train PathCNN with one stratified split: 70% train, 10% val, 20% test."""
    print("Training PathCNN with single split (70% train / 10% val / 20% test)...")

    results_dir = "results_single_split"
    os.makedirs(results_dir, exist_ok=True)

    batch_size = 64
    min_epochs = 50
    max_epochs = 200
    patience = 25
    num_classes = 2

    img_rows, img_cols = pathway_images.shape[1], pathway_images.shape[2]
    input_shape = (img_rows, img_cols, 1)

    X = pathway_images.reshape(pathway_images.shape[0], img_rows, img_cols, 1).astype("float32")
    y = labels
    sample_ids = np.arange(len(y))

    X_train, X_temp, y_train, y_temp, id_train, id_temp = train_test_split(
        X,
        y,
        sample_ids,
        test_size=0.30,
        random_state=random_state,
        stratify=y,
    )

    X_val, X_test, y_val, y_test, id_val, id_test = train_test_split(
        X_temp,
        y_temp,
        id_temp,
        test_size=2.0 / 3.0,
        random_state=random_state,
        stratify=y_temp,
    )

    print(f"Input shape: {X.shape}")
    print(f"Train size: {len(X_train)} ({len(X_train) / len(X) * 100:.1f}%)")
    print(f"Val size: {len(X_val)} ({len(X_val) / len(X) * 100:.1f}%)")
    print(f"Test size: {len(X_test)} ({len(X_test) / len(X) * 100:.1f}%)")
    print(f"Train label distribution: {np.bincount(y_train)}")
    print(f"Val label distribution: {np.bincount(y_val)}")
    print(f"Test label distribution: {np.bincount(y_test)}")

    dummy_train = np.zeros((len(X_train), 1))
    dummy_val = np.zeros((len(X_val), 1))
    dummy_test = np.zeros((len(X_test), 1))

    y_train_cat = keras.utils.to_categorical(y_train, num_classes)
    y_val_cat = keras.utils.to_categorical(y_val, num_classes)
    y_test_cat = keras.utils.to_categorical(y_test, num_classes)

    model = create_pathcnn_model(input_shape, num_classes)
    optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    class_counts = np.bincount(y_train)
    class_weight = {0: 1.0, 1: class_counts[0] / class_counts[1]} if len(class_counts) > 1 and class_counts[1] > 0 else {0: 1.0}
    print(f"Class weights: {class_weight}")

    early_stopping = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1)

    history = model.fit(
        [X_train, dummy_train],
        y_train_cat,
        batch_size=batch_size,
        epochs=max_epochs,
        verbose=1,
        class_weight=class_weight,
        validation_data=([X_val, dummy_val], y_val_cat),
        callbacks=[early_stopping],
    )

    epochs_trained = len(history.history["loss"])
    if epochs_trained < min_epochs:
        print(f"Training additional {min_epochs - epochs_trained} epochs to reach minimum...")
        model.fit(
            [X_train, dummy_train],
            y_train_cat,
            batch_size=batch_size,
            epochs=min_epochs - epochs_trained,
            verbose=1,
            class_weight=class_weight,
            validation_data=([X_val, dummy_val], y_val_cat),
        )
        epochs_trained = min_epochs

    def evaluate_split(split_name, X_split, y_split, dummy_split):
        y_prob = model.predict([X_split, dummy_split], verbose=0)[:, 1]
        y_bin = (y_prob >= 0.5).astype(int)
        m = calculate_metrics(y_split, y_prob, y_bin)
        print(
            f"{split_name:<5s} | ACC={m['accuracy']:.4f}, AUC={m['auc']:.4f}, AUPR={m['aupr']:.4f}, "
            f"F1={m['f1']:.4f}, Precision={m['precision']:.4f}, Recall={m['recall']:.4f}"
        )
        return m

    print("\nFinal metrics:")
    train_metrics = evaluate_split("Train", X_train, y_train, dummy_train)
    val_metrics = evaluate_split("Valid", X_val, y_val, dummy_val)
    test_metrics = evaluate_split("Test", X_test, y_test, dummy_test)

    feature_model = Model(inputs=model.inputs, outputs=model.get_layer("embedding_dense").output)
    train_embeddings = feature_model.predict([X_train, dummy_train], verbose=0)
    test_embeddings = feature_model.predict([X_test, dummy_test], verbose=0)

    embedding_outdir = os.path.join(results_dir, "embedding_separation")
    os.makedirs(embedding_outdir, exist_ok=True)

    train_sep_summary = analyze_embedding_separation(
        embeddings=train_embeddings,
        y_true=y_train,
        sample_ids=id_train,
        outdir=embedding_outdir,
        split_name="train",
        random_state=random_state,
    )
    test_sep_summary = analyze_embedding_separation(
        embeddings=test_embeddings,
        y_true=y_test,
        sample_ids=id_test,
        outdir=embedding_outdir,
        split_name="test",
        random_state=random_state,
    )

    final_results = {
        "experiment_info": {
            "split": "70/10/20 (train/val/test)",
            "random_state": random_state,
            "min_epochs": min_epochs,
            "max_epochs": max_epochs,
            "patience": patience,
            "batch_size": batch_size,
            "input_shape": list(X.shape),
            "num_classes": num_classes,
            "total_samples": len(X),
            "label_distribution": np.bincount(y).tolist(),
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
            "train_distribution": np.bincount(y_train).tolist(),
            "val_distribution": np.bincount(y_val).tolist(),
            "test_distribution": np.bincount(y_test).tolist(),
            "epochs_trained": epochs_trained,
        },
        "metrics": {
            "train": train_metrics,
            "valid": val_metrics,
            "test": test_metrics,
        },
        "embedding_separation": {
            "train": train_sep_summary,
            "test": test_sep_summary,
        },
    }

    json_path = os.path.join(results_dir, "pathcnn_single_split_results.json")
    with open(json_path, "w") as f:
        json.dump(final_results, f, indent=4)

    metrics_df = pd.DataFrame(final_results["metrics"]).T
    metrics_df.to_csv(os.path.join(results_dir, "single_split_metrics.csv"), index_label="split")

    print("\nSaved outputs:")
    print(f"  - {json_path}")
    print(f"  - {os.path.join(results_dir, 'single_split_metrics.csv')}")
    print(f"  - {embedding_outdir}/train/* and {embedding_outdir}/test/*")

    K.clear_session()
    return final_results


def main():
    """Run PathCNN pipeline with one 70/10/20 split and UMAP visualization."""
    print("Starting PathCNN pipeline with one split and UMAP visualization...")

    mutation_data, cnv_data, labels, gene_names, pathway_genes = load_and_preprocess_data()
    pathway_mapping, _ = create_pathway_gene_mapping(pathway_genes, gene_names)

    mutation_pca, _ = apply_pca_to_pathways(mutation_data, pathway_mapping, gene_names)
    cnv_pca, _ = apply_pca_to_pathways(cnv_data, pathway_mapping, gene_names)

    pathway_images = create_pathway_images(mutation_pca, cnv_pca, n_pc=2)
    ordered_pathway_images, _ = order_pathways_by_correlation(pathway_images)

    print(f"Final pathway images shape: {ordered_pathway_images.shape}")

    final_results = train_pathcnn_single_split(ordered_pathway_images, labels, random_state=42)

    print("\nPathCNN single-split pipeline completed.")
    return final_results


if __name__ == "__main__":
    main()
