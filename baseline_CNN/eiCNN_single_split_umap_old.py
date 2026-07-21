import copy
import gc
import json
import os
import random
import warnings

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    pairwise_distances,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

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


def setup_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class EarlyIntegrationCNN(torch.nn.Module):
    """Early Integration CNN baseline architecture."""

    def __init__(self, in_dim, num_classes):
        super(EarlyIntegrationCNN, self).__init__()
        conv1_out = in_dim - 1000 + 1
        pool1_out = conv1_out // 100
        conv2_out = pool1_out - 50 + 1
        pool2_out = conv2_out // 10
        linear_input = pool2_out * 16

        self.FC = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=1000),
            nn.ReLU(),
            nn.MaxPool1d(100),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=50),
            nn.ReLU(),
            nn.MaxPool1d(10),
            nn.Flatten(),
            nn.Linear(in_features=int(linear_input), out_features=50),
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=num_classes),
        )
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, num_classes):
        x = x.unsqueeze(1)
        x = self.FC(x)
        if num_classes == 2:
            dec_logits = self.sigmoid(x)
        else:
            dec_logits = self.softmax(x)
        return dec_logits


class MultiOmicsDataset(Dataset):
    """Dataset wrapper for integrated omics matrix and labels."""

    def __init__(self, data, label):
        super().__init__()
        self.data = data
        self.label = label

    def __getitem__(self, index):
        full_seq = torch.from_numpy(self.data[index]).float()
        seq_label = self.label[index]
        return full_seq, seq_label

    def __len__(self):
        return self.data.shape[0]


class EarlyStopping:
    """Early stopping on monitored metric."""

    def __init__(self, patience=25, verbose=False, delta=0, stop=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        self.save_epoch = True
        self.delta = delta
        self.stop = stop

    def __call__(self, monitor, epoch):
        score = monitor[0] if len(monitor) == 1 else np.mean(monitor)

        if self.best_epoch is None:
            self.best_epoch = epoch

        if epoch <= self.stop:
            self.best_score = score
            self.early_stop = False
            self.best_epoch = epoch
            self.counter = 0

        if (self.best_score is None) or (epoch == 1):
            self.best_score = score
            self.save_epoch = True
        elif score < self.best_score - self.delta:
            self.counter += 1
            self.save_epoch = False
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.save_epoch = True
            self.best_epoch = epoch


def calculate_comprehensive_metrics(y_true, y_pred_proba, y_pred_binary):
    """Calculate comprehensive binary metrics."""
    metrics = {}

    try:
        metrics["AUC_ROC"] = roc_auc_score(y_true, y_pred_proba)
    except Exception:
        metrics["AUC_ROC"] = np.nan

    try:
        metrics["AUC_PR"] = average_precision_score(y_true, y_pred_proba)
    except Exception:
        metrics["AUC_PR"] = np.nan

    metrics["Accuracy"] = accuracy_score(y_true, y_pred_binary)
    metrics["F1_Binary"] = f1_score(y_true, y_pred_binary, average="binary", zero_division=0)
    metrics["F1_Weighted"] = f1_score(y_true, y_pred_binary, average="weighted", zero_division=0)
    metrics["F1_Macro"] = f1_score(y_true, y_pred_binary, average="macro", zero_division=0)
    metrics["Precision_Binary"] = precision_score(y_true, y_pred_binary, average="binary", zero_division=0)
    metrics["Recall_Binary"] = recall_score(y_true, y_pred_binary, average="binary", zero_division=0)
    metrics["Precision"] = precision_score(y_true, y_pred_binary, average="weighted", zero_division=0)
    metrics["Recall"] = recall_score(y_true, y_pred_binary, average="weighted", zero_division=0)

    return metrics


def load_and_prepare_data():
    """Load mutation/CNV/labels and build early-integrated features."""
    print("Loading multi-omics data...")

    mutation_data = pd.read_csv("mutation_data.csv")
    cnv_data = pd.read_csv("cnv_data.csv")
    labels_data = pd.read_csv("labels.csv")

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

    gene_names = mutation_aligned.columns[1:].tolist()
    mutation_matrix = mutation_aligned.iloc[:, 1:].values.astype(float)
    cnv_matrix = cnv_aligned.iloc[:, 1:].values.astype(float)
    labels = labels_aligned.iloc[:, 1].values.astype(int)
    patient_ids = mutation_aligned.iloc[:, 0].values

    integrated_data = np.concatenate([mutation_matrix, cnv_matrix], axis=1)

    print(f"Integrated data shape: {integrated_data.shape}")
    print(f"Number of genes: {len(gene_names)}")
    print(f"Label distribution: {np.bincount(labels)}")

    return integrated_data, labels, gene_names, patient_ids


def _safe_2d_projection(embeddings, method="umap", random_state=42):
    """Project embeddings to 2D; UMAP -> TSNE -> PCA fallback."""
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

    if method.lower() == "umap":
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
            pass

    try:
        perplexity = max(2, min(30, n_samples - 1))
        proj = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=random_state,
            init="pca",
            learning_rate="auto",
        ).fit_transform(emb_in)
        return proj, "tsne"
    except Exception:
        return PCA(n_components=2, random_state=random_state).fit_transform(embeddings), "pca"


def analyze_embedding_separation(embeddings, y_true, sample_ids, outdir, split_name, random_state=42):
    """Create projection and class-separation distance plots."""
    if embeddings.size == 0 or len(y_true) == 0:
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


def _collect_probs_and_labels(model, loader, device, num_classes=2):
    """Collect positive-class probabilities and true labels from a loader."""
    model.eval()
    probs = []
    labels = []
    pin_memory = device.type == "cuda"

    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device, non_blocking=pin_memory)
            targets = targets.to(device, non_blocking=pin_memory)
            targets = targets[:, 0] if len(targets.shape) > 1 else targets
            outputs = model(data, num_classes=num_classes)
            probs.extend(outputs[:, 1].detach().cpu().numpy())
            labels.extend(targets.detach().cpu().numpy())

    probs = np.array(probs)
    labels = np.array(labels).astype(int)
    return probs, labels


def _collect_embeddings(model, loader, device):
    """Collect penultimate embeddings (50-d) from FC stack before final classifier."""
    model.eval()
    core_model = model.module if isinstance(model, nn.DataParallel) else model
    embeddings = []

    pin_memory = device.type == "cuda"
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device, non_blocking=pin_memory)
            x = data.unsqueeze(1)
            emb = core_model.FC[:-1](x)
            embeddings.append(emb.detach().cpu().numpy())

    if not embeddings:
        return np.empty((0, 0))
    return np.concatenate(embeddings, axis=0)


def train_early_integration_cnn_single_split(
    integrated_data,
    labels,
    patient_ids,
    output_dir="early_integration_cnn_single_split_results",
    random_state=42,
):
    """Train early-integration CNN with one split: 70% train / 10% valid / 20% test."""
    print("Training Early Integration CNN with single split (70/10/20)...")

    os.makedirs(output_dir, exist_ok=True)

    batch_size = 32
    epochs = 100
    learning_rate = 0.001
    patience = 25
    delta = 0.001
    stop_epoch = 10
    num_classes = 2
    grad_acc = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    setup_seed(2022)

    indices = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.30,
        random_state=random_state,
        stratify=labels,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=2.0 / 3.0,
        random_state=random_state,
        stratify=labels[temp_idx],
    )

    X_train_raw = integrated_data[train_idx]
    X_val_raw = integrated_data[val_idx]
    X_test_raw = integrated_data[test_idx]
    y_train = labels[train_idx]
    y_val = labels[val_idx]
    y_test = labels[test_idx]

    id_train = patient_ids[train_idx]
    id_val = patient_ids[val_idx]
    id_test = patient_ids[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    print(f"Input shape: {integrated_data.shape}")
    print(f"Train size: {len(train_idx)} ({len(train_idx) / len(labels) * 100:.1f}%)")
    print(f"Valid size: {len(val_idx)} ({len(val_idx) / len(labels) * 100:.1f}%)")
    print(f"Test size: {len(test_idx)} ({len(test_idx) / len(labels) * 100:.1f}%)")
    print(f"Train label distribution: {np.bincount(y_train)}")
    print(f"Valid label distribution: {np.bincount(y_val)}")
    print(f"Test label distribution: {np.bincount(y_test)}")

    y_train_tensor = torch.LongTensor(y_train)
    y_val_tensor = torch.LongTensor(y_val)
    y_test_tensor = torch.LongTensor(y_test)

    train_dataset = MultiOmicsDataset(X_train, y_train_tensor)
    val_dataset = MultiOmicsDataset(X_val, y_val_tensor)
    test_dataset = MultiOmicsDataset(X_test, y_test_tensor)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    model = EarlyIntegrationCNN(in_dim=integrated_data.shape[1], num_classes=num_classes)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    class_counts = np.bincount(y_train)
    if len(class_counts) < 2 or class_counts[1] == 0:
        class_weight = torch.tensor([1.0, 1.0], dtype=torch.float32).to(device)
    else:
        class_weight = torch.tensor(
            [sum(class_counts) / (2 * class_counts[0]), sum(class_counts) / (2 * class_counts[1])],
            dtype=torch.float32,
        ).to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss(weight=class_weight)
    early_stopping = EarlyStopping(patience=patience, verbose=False, delta=delta, stop=stop_epoch)

    best_state_dict = copy.deepcopy(model.state_dict())
    best_epoch = 1

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, (data, targets) in enumerate(train_loader, start=1):
            data = data.to(device, non_blocking=pin_memory)
            targets = targets.to(device, non_blocking=pin_memory)
            targets = targets[:, 0] if len(targets.shape) > 1 else targets

            outputs = model(data, num_classes=num_classes)
            loss = loss_fn(outputs, targets)
            loss.backward()

            if batch_idx % grad_acc == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e6))
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()

        if len(train_loader) % grad_acc != 0:
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        y_val_prob, y_val_true = _collect_probs_and_labels(model, val_loader, device, num_classes=num_classes)
        y_val_pred = (y_val_prob > 0.5).astype(int)
        val_metrics = calculate_comprehensive_metrics(y_val_true, y_val_prob, y_val_pred)
        val_f1_macro = val_metrics["F1_Macro"]

        early_stopping([val_f1_macro], epoch)
        if early_stopping.save_epoch:
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch

        if early_stopping.early_stop and epoch > stop_epoch:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state_dict)
    print(f"Best model restored from epoch {best_epoch}")

    y_train_prob, y_train_true = _collect_probs_and_labels(model, train_loader, device, num_classes=num_classes)
    y_val_prob, y_val_true = _collect_probs_and_labels(model, val_loader, device, num_classes=num_classes)
    y_test_prob, y_test_true = _collect_probs_and_labels(model, test_loader, device, num_classes=num_classes)

    y_train_pred = (y_train_prob > 0.5).astype(int)
    y_val_pred = (y_val_prob > 0.5).astype(int)
    y_test_pred = (y_test_prob > 0.5).astype(int)

    train_metrics = calculate_comprehensive_metrics(y_train_true, y_train_prob, y_train_pred)
    valid_metrics = calculate_comprehensive_metrics(y_val_true, y_val_prob, y_val_pred)
    test_metrics = calculate_comprehensive_metrics(y_test_true, y_test_prob, y_test_pred)

    print("\nFinal metrics:")
    print(
        "Train | "
        f"ACC={train_metrics['Accuracy']:.4f}, AUC={train_metrics['AUC_ROC']:.4f}, "
        f"AUPR={train_metrics['AUC_PR']:.4f}, F1={train_metrics['F1_Binary']:.4f}, "
        f"Precision={train_metrics['Precision_Binary']:.4f}, Recall={train_metrics['Recall_Binary']:.4f}"
    )
    print(
        "Valid | "
        f"ACC={valid_metrics['Accuracy']:.4f}, AUC={valid_metrics['AUC_ROC']:.4f}, "
        f"AUPR={valid_metrics['AUC_PR']:.4f}, F1={valid_metrics['F1_Binary']:.4f}, "
        f"Precision={valid_metrics['Precision_Binary']:.4f}, Recall={valid_metrics['Recall_Binary']:.4f}"
    )
    print(
        "Test  | "
        f"ACC={test_metrics['Accuracy']:.4f}, AUC={test_metrics['AUC_ROC']:.4f}, "
        f"AUPR={test_metrics['AUC_PR']:.4f}, F1={test_metrics['F1_Binary']:.4f}, "
        f"Precision={test_metrics['Precision_Binary']:.4f}, Recall={test_metrics['Recall_Binary']:.4f}"
    )

    train_embeddings = _collect_embeddings(model, train_loader, device)
    test_embeddings = _collect_embeddings(model, test_loader, device)

    embedding_outdir = os.path.join(output_dir, "embedding_separation")
    os.makedirs(embedding_outdir, exist_ok=True)

    train_sep_summary = analyze_embedding_separation(
        embeddings=train_embeddings,
        y_true=y_train_true,
        sample_ids=id_train,
        outdir=embedding_outdir,
        split_name="train",
        random_state=random_state,
    )
    test_sep_summary = analyze_embedding_separation(
        embeddings=test_embeddings,
        y_true=y_test_true,
        sample_ids=id_test,
        outdir=embedding_outdir,
        split_name="test",
        random_state=random_state,
    )

    preds_df = pd.DataFrame(
        {
            "sample": id_test,
            "y_label": y_test_true.astype(int),
            "y_score": y_test_prob,
            "y_pred": y_test_pred,
        }
    ).sort_values("sample")
    preds_csv = os.path.join(output_dir, "early_integration_cnn_test_predictions.csv")
    preds_df.to_csv(preds_csv, index=False)

    metrics_df = pd.DataFrame({"Train": train_metrics, "Valid": valid_metrics, "Test": test_metrics}).T
    metrics_csv = os.path.join(output_dir, "early_integration_cnn_single_split_metrics.csv")
    metrics_df.to_csv(metrics_csv, index_label="split")

    final_results = {
        "experiment_info": {
            "split": "70/10/20 (train/val/test)",
            "random_state": random_state,
            "epochs_max": epochs,
            "patience": patience,
            "best_epoch": best_epoch,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "total_samples": int(len(labels)),
            "label_distribution": np.bincount(labels).tolist(),
            "train_size": int(len(train_idx)),
            "val_size": int(len(val_idx)),
            "test_size": int(len(test_idx)),
            "train_distribution": np.bincount(y_train).tolist(),
            "val_distribution": np.bincount(y_val).tolist(),
            "test_distribution": np.bincount(y_test).tolist(),
        },
        "metrics": {"train": train_metrics, "valid": valid_metrics, "test": test_metrics},
        "embedding_separation": {"train": train_sep_summary, "test": test_sep_summary},
        "artifacts": {
            "metrics_csv": metrics_csv,
            "test_predictions_csv": preds_csv,
            "embedding_dir": embedding_outdir,
        },
    }

    results_json = os.path.join(output_dir, "early_integration_cnn_single_split_results.json")
    with open(results_json, "w") as f:
        json.dump(final_results, f, indent=2)

    print("\nSaved outputs:")
    print(f"  - {results_json}")
    print(f"  - {metrics_csv}")
    print(f"  - {preds_csv}")
    print(f"  - {embedding_outdir}/train/* and {embedding_outdir}/test/*")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return final_results


def main():
    """Main pipeline for single-split early integration CNN + UMAP analysis."""
    print("Starting Early Integration CNN single-split pipeline...")

    output_dir = "early_integration_cnn_single_split_results"
    os.makedirs(output_dir, exist_ok=True)

    integrated_data, labels, _, patient_ids = load_and_prepare_data()
    train_early_integration_cnn_single_split(
        integrated_data=integrated_data,
        labels=labels,
        patient_ids=patient_ids,
        output_dir=output_dir,
        random_state=42,
    )

    print("\nEarly Integration CNN single-split pipeline completed.")


if __name__ == "__main__":
    main()
