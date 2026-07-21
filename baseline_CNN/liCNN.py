import os
import gc
import json
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
                             average_precision_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.optim import AdamW
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import warnings
warnings.filterwarnings('ignore')

def setup_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Favor GPU throughput for CNN kernels.
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

class LateIntegrationCNN(torch.nn.Module):
    """
    Late Integration CNN - same architecture as liCNN baseline
    """
    def __init__(self, in_dim, num_classes):
        super(LateIntegrationCNN, self).__init__()
        
        # Calculate output dimensions for each CNN branch
        # For each branch: (input_size - 300 + 1) / 100
        mut_conv_out = (in_dim[0] - 300 + 1) // 100
        cnv_conv_out = (in_dim[1] - 300 + 1) // 100
        
        # Separate CNN branches for each omics type
        self.FC_1 = nn.Sequential(  # Mutation CNN
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=300),
            nn.ReLU(),
            nn.MaxPool1d(100),
            nn.Flatten()
        )
        
        self.FC_2 = nn.Sequential(  # CNV CNN
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=300),
            nn.ReLU(),
            nn.MaxPool1d(100),
            nn.Flatten()
        )
        
        # Merged network
        total_features = int(mut_conv_out * 32 + cnv_conv_out * 32)
        self.FC_merge = nn.Sequential(
            nn.Linear(in_features=total_features, out_features=100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=100, out_features=50),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=50, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=num_classes)
        )
        
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, in_dim, num_classes):
        # Split input into different omics types
        x_mut = x[:, :in_dim[0]].unsqueeze(1)  # Mutation data with channel dim
        x_cnv = x[:, in_dim[0]:in_dim[0]+in_dim[1]].unsqueeze(1)  # CNV data with channel dim
        
        # Process each omics type separately
        x_1 = self.FC_1(x_mut)  # Mutation CNN
        x_2 = self.FC_2(x_cnv)  # CNV CNN
        
        # Concatenate the outputs
        x = torch.cat((x_1, x_2), 1)
        
        # Final merged processing
        x = self.FC_merge(x)
        
        if num_classes == 2:
            dec_logits = self.sigmoid(x)
        else:
            dec_logits = self.softmax(x)
        
        return dec_logits

class MultiOmicsDataset(Dataset):
    """Custom Dataset for multi-omics data"""
    def __init__(self, data, label):
        super().__init__()
        if isinstance(data, np.ndarray):
            self.data = torch.from_numpy(data).float()
        else:
            self.data = data.float()
        if isinstance(label, np.ndarray):
            self.label = torch.from_numpy(label).long()
        else:
            self.label = label.long()

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.data.shape[0]

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=25, verbose=False, delta=0, stop=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        self.delta = delta
        self.stop = stop

    def __call__(self, monitor, epoch):
        if len(monitor) == 1:
            score = monitor[0]
        else:
            score = np.mean(monitor)
        
        if self.best_epoch is None:
            self.best_epoch = epoch
        
        if epoch <= self.stop:
            self.best_score = score
            self.early_stop = False
            self.best_epoch = epoch
            self.counter = 0

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                print(f'EarlyStopping best_epoch: {self.best_epoch}')
        else:
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch

def compute_metrics_with_best_f1_threshold(y_true, y_scores):
    """
    Compute downstream metrics using a fixed 0.5 threshold.
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    best_thr = 0.5

    y_pred = (y_scores >= best_thr).astype(int)
    auc_val = roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else float("nan")
    aupr_val = average_precision_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else float("nan")
    acc_val = accuracy_score(y_true, y_pred)
    f1_bin = f1_score(y_true, y_pred, average='binary', zero_division=0)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)

    return {
        "acc": acc_val,
        "auc": auc_val,
        "aupr": aupr_val,
        "f1_binary": f1_bin,
        "precision": precision,
        "recall": recall,
        "best_threshold": best_thr,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_probs": y_scores
    }

def get_roc(y, pro):
    """
    Calculate binary classification metrics using a fixed 0.5 threshold.
    """
    base_metrics = compute_metrics_with_best_f1_threshold(y, pro)
    f1_macro = f1_score(base_metrics["y_true"], base_metrics["y_pred"], average='macro', zero_division=0)
    return {
        "acc": base_metrics["acc"],
        "auc": base_metrics["auc"],
        "aupr": base_metrics["aupr"],
        "f1_binary": base_metrics["f1_binary"],
        "f1_macro": f1_macro,
        "precision": base_metrics["precision"],
        "recall": base_metrics["recall"],
        "best_threshold": base_metrics["best_threshold"],
        "y_true": base_metrics["y_true"],
        "y_pred": base_metrics["y_pred"],
        "y_probs": base_metrics["y_probs"]
    }

def calculate_comprehensive_metrics(y_true, y_pred_proba):
    """
    Calculate metrics using a fixed 0.5 threshold.
    """
    base_metrics = compute_metrics_with_best_f1_threshold(y_true, y_pred_proba)
    f1_macro = f1_score(base_metrics["y_true"], base_metrics["y_pred"], average='macro', zero_division=0)

    metrics = {
        'AUC_ROC': base_metrics['auc'],
        'AUC_PR': base_metrics['aupr'],
        'Accuracy': base_metrics['acc'],
        'F1_Binary': base_metrics['f1_binary'],
        'F1_Macro': f1_macro,
        'Precision': base_metrics['precision'],
        'Recall': base_metrics['recall'],
        'best_threshold': base_metrics['best_threshold'],
        'y_true': base_metrics['y_true'],
        'y_pred': base_metrics['y_pred'],
        'y_probs': base_metrics['y_probs']
    }
    
    return metrics

def load_and_prepare_data():
    """Load and prepare multi-omics data for late integration CNN"""
    print("Loading multi-omics data...")
    
    # Load data files
    mutation_data = pd.read_csv("mutation_data.csv")
    cnv_data = pd.read_csv("cnv_data.csv")
    labels_data = pd.read_csv("labels.csv")
    
    # Extract patient IDs and align datasets
    mutation_patients = mutation_data.iloc[:, 0].values
    cnv_patients = cnv_data.iloc[:, 0].values
    label_patients = labels_data.iloc[:, 0].values
    
    # Find common patients across all datasets
    common_patients = list(set(mutation_patients) & set(cnv_patients) & set(label_patients))
    print(f"Found {len(common_patients)} common patients across all datasets")
    
    # Align datasets by common patients
    mutation_aligned = mutation_data[mutation_data.iloc[:, 0].isin(common_patients)].reset_index(drop=True)
    cnv_aligned = cnv_data[cnv_data.iloc[:, 0].isin(common_patients)].reset_index(drop=True)
    labels_aligned = labels_data[labels_data.iloc[:, 0].isin(common_patients)].reset_index(drop=True)
    
    # Sort by patient ID to ensure alignment
    mutation_aligned = mutation_aligned.sort_values(by=mutation_aligned.columns[0]).reset_index(drop=True)
    cnv_aligned = cnv_aligned.sort_values(by=cnv_aligned.columns[0]).reset_index(drop=True)
    labels_aligned = labels_aligned.sort_values(by=labels_aligned.columns[0]).reset_index(drop=True)
    
    # Extract gene names and data matrices
    gene_names = mutation_aligned.columns[1:].tolist()
    mutation_matrix = mutation_aligned.iloc[:, 1:].values.astype(float)
    cnv_matrix = cnv_aligned.iloc[:, 1:].values.astype(float)
    labels = labels_aligned.iloc[:, 1].values.astype(int)
    patient_ids = mutation_aligned.iloc[:, 0].values
    
    # Verify data integrity
    assert mutation_matrix.shape[0] == cnv_matrix.shape[0] == len(labels), \
        "Mismatch in number of samples across datasets"
    
    print(f"Mutation data shape: {mutation_matrix.shape}")
    print(f"CNV data shape: {cnv_matrix.shape}")
    print(f"Number of samples: {len(labels)}")
    print(f"Number of genes: {len(gene_names)}")
    
    # Late integration: Keep separate, concatenate later
    integrated_data = np.concatenate([mutation_matrix, cnv_matrix], axis=1)
    feature_dims = (mutation_matrix.shape[1], cnv_matrix.shape[1])
    
    print(f"Integrated data shape: {integrated_data.shape}")
    print(f"Feature dimensions (mutation, CNV): {feature_dims}")
    
    # Label distribution
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nLabel distribution:")
    for label, count in zip(unique, counts):
        print(f"Class {label}: {count} samples ({count/len(labels)*100:.1f}%)")
    
    return integrated_data, labels, feature_dims, gene_names, patient_ids

def train_late_integration_cnn(integrated_data, labels, feature_dims, patient_ids, n_folds=5, output_dir="late_integration_cnn_results"):
    """
    Train Late Integration CNN with stratified cross-validation

    Cross-Validation Setup:
    - 5-fold stratified cross-validation (n_folds=5)
    - Each fold: ~20% test, ~80% train_val
    - From 80% train_val: val_size = 0.10 → ~8% validation, ~72% training
    - random_state=42 for reproducibility
    - stratify=True to maintain class distribution

    Final data split per fold:
    - Training:   ~72% of total data
    - Validation: ~8% of total data
    - Test:       ~20% of total data
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    else:
        print("WARNING: CUDA not available, using CPU\n")
    
    # Parameters aligned with main_soft_masking_2modal.py (SKILLS.md)
    batch_size = 16
    epochs = 200
    learning_rate = 0.0001
    weight_decay = 5e-4
    patience = 25
    delta = 0.001
    stop_epoch = 50
    num_classes = 2
    grad_acc = 1
    use_amp = device.type == 'cuda'
    amp_scaler = GradScaler(enabled=use_amp)
    pin_memory = device.type == 'cuda'
    num_workers = min(8, os.cpu_count() or 1)
    
    # Setup - CHANGED: Using seed 42 for reproducibility
    seed = 42
    setup_seed(seed)
    
    # Data preprocessing
    data_scaler = StandardScaler()
    integrated_data_scaled = data_scaler.fit_transform(integrated_data)
    
    print(f'Input shape: {integrated_data_scaled.shape}')
    print(f'Labels shape: {labels.shape}')
    print(f'Feature dimensions: {feature_dims}')
    
    # Initialize metric storage for each fold - CHANGED: F1_Binary instead of F1_Weighted
    fold_metrics_list = {
        'AUC_ROC': [], 'AUC_PR': [], 'Accuracy': [], 
        'F1_Binary': [], 'F1_Macro': [], 'Precision': [], 'Recall': []
    }
    
    print(f"\n{'='*60}")
    print(f"Starting {n_folds}-Fold Cross-Validation")
    print(f"CV Setup: {n_folds}-fold, ~20% test per fold, ~10% val from train (72/8/20 total), random_state=42")
    print(f"{'='*60}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Stratified K-Fold with fixed random_state=42
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_fold_predictions = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(integrated_data_scaled, labels)):
        print(f"\nFold {fold + 1}/{n_folds}")
        print(f"{'-'*60}")
        
        # Split data: 80% train, 20% test (from outer CV)
        X_train, X_test = integrated_data_scaled[train_idx], integrated_data_scaled[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        patient_train, patient_test = patient_ids[train_idx], patient_ids[test_idx]
        
        # Stratified split: val = 10% of the 80% train_val pool, giving
        # per-fold ~72% train / ~8% val / ~20% test (5-fold 80/20 outer +
        # 10%-of-train inner val).
        X_train_final, X_val, y_train_final, y_val, patient_train_final, patient_val = train_test_split(
            X_train, y_train, patient_train,
            test_size=0.10,
            random_state=42,
            stratify=y_train
        )
        
        print(f"Train set: {len(X_train_final)} samples (Class distribution: {np.bincount(y_train_final)})")
        print(f"Val set:   {len(X_val)} samples (Class distribution: {np.bincount(y_val)})")
        print(f"Test set:  {len(X_test)} samples (Class distribution: {np.bincount(y_test)})")
        
        # Convert to tensors
        y_train_tensor = torch.LongTensor(y_train_final)
        y_val_tensor = torch.LongTensor(y_val)
        y_test_tensor = torch.LongTensor(y_test)
        
        # Create datasets and data loaders
        train_dataset = MultiOmicsDataset(X_train_final, y_train_tensor)
        val_dataset = MultiOmicsDataset(X_val, y_val_tensor)
        test_dataset = MultiOmicsDataset(X_test, y_test_tensor)

        loader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 4

        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
        test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
        
        # Create model and move to device
        model = LateIntegrationCNN(in_dim=feature_dims, num_classes=num_classes)
        model = model.to(device)
        
        # Calculate class weights for imbalanced data and move to device
        class_counts = np.bincount(y_train_final)
        class_weight = torch.tensor([sum(class_counts) / (2 * x) for x in class_counts], dtype=torch.float32)
        class_weight = class_weight.to(device)
        
        # Optimizer and loss function
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        loss_fn = nn.CrossEntropyLoss(weight=class_weight)
        early_stopping = EarlyStopping(patience=patience, verbose=False, delta=delta, stop=stop_epoch)
        
        # Training loop
        for epoch in range(1, epochs + 1):
            # Training phase
            model.train()
            running_loss = torch.zeros((), device=device)
            y_train_pred = []
            y_train_true = []
            num_train_batches = len(train_loader)
            optimizer.zero_grad(set_to_none=True)

            for batch_idx, (data, targets) in enumerate(train_loader, start=1):
                # Move data and targets to device
                data = data.to(device, non_blocking=pin_memory)
                targets = targets.to(device, non_blocking=pin_memory)
                targets = targets[:, 0] if len(targets.shape) > 1 else targets

                with autocast(enabled=use_amp):
                    outputs = model(data, in_dim=feature_dims, num_classes=num_classes)
                    loss = loss_fn(outputs, targets)
                scaled_loss = loss / grad_acc
                amp_scaler.scale(scaled_loss).backward()

                if batch_idx % grad_acc == 0 or batch_idx == num_train_batches:
                    amp_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e6))
                    amp_scaler.step(optimizer)
                    amp_scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                running_loss += loss.detach().float()
                y_train_pred.append(outputs[:, 1].detach())
                y_train_true.append(targets.detach())

            epoch_loss = (running_loss / max(1, num_train_batches)).item()
            y_train_pred_np = torch.cat(y_train_pred, dim=0).float().cpu().numpy()
            y_train_true_np = torch.cat(y_train_true, dim=0).cpu().numpy()
            # Metrics using PR-curve best-F1 threshold (aligned with main_soft_masking.py)
            train_metrics = get_roc(y_train_true_np, y_train_pred_np)
            ACC_train = train_metrics["acc"]
            AUC_train = train_metrics["auc"]
            f1_binary_train = train_metrics["f1_binary"]
            f1_macro_train = train_metrics["f1_macro"]
            
            # Validation phase
            model.eval()
            val_loss = torch.zeros((), device=device)
            y_val_pred = []
            y_val_true = []
            
            with torch.inference_mode():
                for data, targets in val_loader:
                    # Move data and targets to device
                    data = data.to(device, non_blocking=pin_memory)
                    targets = targets.to(device, non_blocking=pin_memory)
                    targets = targets[:, 0] if len(targets.shape) > 1 else targets

                    with autocast(enabled=use_amp):
                        outputs = model(data, in_dim=feature_dims, num_classes=num_classes)
                        loss = loss_fn(outputs, targets)
                    val_loss += loss.detach().float()
                    y_val_pred.append(outputs[:, 1].detach())
                    y_val_true.append(targets.detach())

            val_loss = (val_loss / max(1, len(val_loader))).item()
            y_val_pred_np = torch.cat(y_val_pred, dim=0).float().cpu().numpy()
            y_val_true_np = torch.cat(y_val_true, dim=0).cpu().numpy()
            # Metrics using PR-curve best-F1 threshold (aligned with main_soft_masking.py)
            val_metrics = get_roc(y_val_true_np, y_val_pred_np)
            ACC_val = val_metrics["acc"]
            AUC_val = val_metrics["auc"]
            f1_binary_val = val_metrics["f1_binary"]
            f1_macro_val = val_metrics["f1_macro"]
            
            # Early stopping check
            early_stopping([f1_macro_val], epoch)
            
            if early_stopping.early_stop and epoch > stop_epoch:
                print(f"  Early stopping at epoch {epoch}")
                break
        
        # Test phase
        model.eval()
        y_test_pred = []
        y_test_true = []
        
        with torch.inference_mode():
            for data, targets in test_loader:
                # Move data and targets to device
                data = data.to(device, non_blocking=pin_memory)
                targets = targets.to(device, non_blocking=pin_memory)
                targets = targets[:, 0] if len(targets.shape) > 1 else targets

                with autocast(enabled=use_amp):
                    outputs = model(data, in_dim=feature_dims, num_classes=num_classes)
                y_test_pred.append(outputs[:, 1].detach())
                y_test_true.append(targets.detach())
        
        # Calculate test metrics using best F1 threshold from PR curve
        y_test_pred = torch.cat(y_test_pred, dim=0).float().cpu().numpy()
        y_test_true = torch.cat(y_test_true, dim=0).cpu().numpy()
        fold_metrics = calculate_comprehensive_metrics(y_test_true, y_test_pred)

        # Save per-sample predictions for this fold
        fold_pred_df = pd.DataFrame({
            'fold': fold + 1,
            'fold_row': np.arange(1, len(patient_test) + 1),
            'Sample': patient_test,
            'y_true_value': y_test_true,
            'y_prediction_score': y_test_pred
        })
        fold_pred_df.index = np.arange(1, len(fold_pred_df) + 1)
        fold_pred_path = os.path.join(output_dir, f'fold_{fold+1}_test_predictions.csv')
        fold_pred_df.to_csv(fold_pred_path)
        all_fold_predictions.append(fold_pred_df.reset_index(drop=True))
        
        # Store fold metrics (only aggregate the core metrics defined above)
        for metric_name, metric_value in fold_metrics.items():
            if metric_name in fold_metrics_list:
                fold_metrics_list[metric_name].append(metric_value)
        
        # CHANGED: Display F1_Binary instead of F1_Weighted
        print(f"\nFold {fold + 1} Test Results:")
        print(f"  AUC-ROC:    {fold_metrics['AUC_ROC']:.4f}")
        print(f"  AUC-PR:     {fold_metrics['AUC_PR']:.4f}")
        print(f"  Accuracy:   {fold_metrics['Accuracy']:.4f}")
        print(f"  F1-Binary:  {fold_metrics['F1_Binary']:.4f}")
        print(f"  F1-Macro:   {fold_metrics['F1_Macro']:.4f}")
        print(f"  Precision:  {fold_metrics['Precision']:.4f}")
        print(f"  Recall:     {fold_metrics['Recall']:.4f}")
        
        # Clear GPU cache after each fold
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
    
    # Calculate mean and std across the 5 folds
    final_stats = {}
    for metric_name, values in fold_metrics_list.items():
        final_stats[f'{metric_name}_mean'] = np.mean(values)
        final_stats[f'{metric_name}_std'] = np.std(values)
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS (Mean ± Std across {n_folds} folds)")
    print(f"{'='*60}")
    # CHANGED: Display F1_Binary instead of F1_Weighted
    for metric in ['AUC_ROC', 'AUC_PR', 'Accuracy', 'F1_Binary', 'F1_Macro', 'Precision', 'Recall']:
        mean_val = final_stats[f'{metric}_mean']
        std_val = final_stats[f'{metric}_std']
        print(f"  {metric:12}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Save combined predictions across all folds (keeping fold info)
    if all_fold_predictions:
        combined_preds = pd.concat(all_fold_predictions, ignore_index=True)
        combined_preds.insert(0, 'id', np.arange(1, len(combined_preds) + 1))
        ordered_cols = ['id', 'fold', 'fold_row', 'Sample', 'y_true_value', 'y_prediction_score']
        combined_preds = combined_preds[[c for c in ordered_cols if c in combined_preds.columns]]
        combined_path = os.path.join(output_dir, 'all_fold_test_predictions.csv')
        combined_preds.to_csv(combined_path, index=False)
        print(f"Combined predictions saved: {combined_path}")
    
    return fold_metrics_list, final_stats

def main():
    """Main function to run Late Integration CNN pipeline"""
    print("="*60)
    print("Late Integration CNN - Stratified 5-Fold Cross-Validation")
    print("="*60)
    
    # Create output directory
    output_dir = "late_integration_cnn_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load and prepare data
    integrated_data, labels, feature_dims, gene_names, patient_ids = load_and_prepare_data()
    
    # Train model with stratified cross-validation
    fold_metrics_list, final_stats = train_late_integration_cnn(
        integrated_data, labels, feature_dims, patient_ids, output_dir=output_dir
    )
    
    # Save detailed results (CSV) - individual fold results
    results_df = pd.DataFrame(fold_metrics_list)
    results_df.index = [f"Fold_{i+1}" for i in range(len(results_df))]
    results_df.to_csv(f"{output_dir}/late_integration_cnn_detailed_results.csv")
    
    # Save summary statistics (CSV)
    summary_df = pd.DataFrame([final_stats])
    summary_df.to_csv(f"{output_dir}/late_integration_cnn_summary_results.csv", index=False)
    
    # Prepare JSON output with mean ± std format
    json_results = {
        "model": "Late Integration CNN",
        "configuration": {
            "n_folds": 5,
            "random_state": 42,
            "stratified": True,
            "validation_split": "10% of train (8% of total)",
            "test_split": "20% of total",
            "metrics": "Binary (positive class focus)"
        },
        "results": {}
    }
    
    # Format results as mean ± std for JSON - CHANGED: F1_Binary instead of F1_Weighted
    for metric in ['AUC_ROC', 'AUC_PR', 'Accuracy', 'F1_Binary', 'F1_Macro', 'Precision', 'Recall']:
        mean_val = final_stats[f'{metric}_mean']
        std_val = final_stats[f'{metric}_std']
        json_results["results"][metric] = {
            "mean": float(mean_val),
            "std": float(std_val),
            "formatted": f"{mean_val:.4f} ± {std_val:.4f}"
        }
    
    # Save JSON file
    json_path = f"{output_dir}/late_integration_cnn_final_results.json"
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=4)

    # Save plain-text summary with mean ± std across folds
    summary_txt = os.path.join(output_dir, "liCNN_cv_summary.txt")
    with open(summary_txt, "w") as f:
        f.write(f"Late Integration CNN - 5-Fold CV Summary (mean ± std across folds)\n")
        f.write(f"n_folds=5, random_state=42\n\n")
        for metric in ['AUC_ROC', 'AUC_PR', 'Accuracy', 'F1_Binary', 'F1_Macro', 'Precision', 'Recall']:
            mean_val = final_stats[f'{metric}_mean']
            std_val = final_stats[f'{metric}_std']
            f.write(f"{metric:12s}: {mean_val:.4f} ± {std_val:.4f}\n")
    
    print(f"\n{'='*60}")
    print(f"Pipeline Completed!")
    print(f"{'='*60}")
    print(f"Detailed results (CSV): {output_dir}/late_integration_cnn_detailed_results.csv")
    print(f"Summary results (CSV):  {output_dir}/late_integration_cnn_summary_results.csv")
    print(f"Final results (JSON):   {json_path}")
    print(f"Text summary:           {summary_txt}")
    print(f"\nJSON results saved with mean ± std format for all metrics")
    print(f"\nNote: All metrics (F1, Precision, Recall) use 'binary' averaging")
    print(f"      This focuses on the positive class (metastatic cancer detection)")

if __name__ == "__main__":
    main()
