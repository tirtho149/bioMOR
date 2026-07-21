import os
import gc
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.optim import AdamW
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import warnings
warnings.filterwarnings('ignore')

def setup_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class EarlyIntegrationCNN(torch.nn.Module):
    """
    Early Integration CNN - same architecture as eiCNN baseline
    """
    def __init__(self, in_dim, num_classes):
        super(EarlyIntegrationCNN, self).__init__()
        # Calculate output dimensions after convolution and pooling
        # After first conv: (in_dim - 1000 + 1) = in_dim - 999
        # After first pool: (in_dim - 999) / 100
        # After second conv: ((in_dim - 999) / 100) - 50 + 1 = ((in_dim - 999) / 100) - 49
        # After second pool: (((in_dim - 999) / 100) - 49) / 10
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
            nn.Linear(in_features=50, out_features=num_classes)
        )
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, num_classes):
        # Reshape for Conv1d: (batch_size, channels, sequence_length)
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.FC(x)
        if num_classes == 2:
            dec_logits = self.sigmoid(x)
        else:
            dec_logits = self.softmax(x)
        return dec_logits

class MultiOmicsDataset(Dataset):
    """Custom Dataset for multi-omics data"""
    def __init__(self, data, label):
        super().__init__()
        self.data = data
        self.label = label

    def __getitem__(self, index):
        full_seq = self.data[index]
        full_seq = torch.from_numpy(full_seq).float()
        seq_label = self.label[index]
        return full_seq, seq_label

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
        self.save_epoch = True
        self.delta = delta
        self.stop = stop

    def __call__(self, monitor, epoch):
        if len(monitor) == 1:
            score = monitor[0]
        else:
            score = np.mean(monitor)
        
        if self.verbose:
            print(f'epoch: {epoch}')
            print(f'Score: {score}')
        
        if self.best_epoch is None:
            self.best_epoch = epoch
        
        if epoch <= self.stop:
            self.best_score = score
            self.early_stop = False
            self.best_epoch = epoch
            self.counter = 0

        if (self.best_score is None) | (epoch == 1):
            self.best_score = score
        elif (score < self.best_score - self.delta):
            self.counter += 1
            self.save_epoch = False
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.save_epoch = True
            self.best_epoch = epoch

def get_roc(y, pro):
    """Calculate binary classification metrics"""
    label = np.array(y)
    pre_label = (pro > 0.5).astype(int)
    auc = roc_auc_score(label, np.array(pro))
    acc = accuracy_score(label, pre_label)
    f1_macro = f1_score(label, pre_label, average='macro')
    f1_weighted = f1_score(label, pre_label, average='weighted')
    return acc, auc, f1_weighted, f1_macro

def calculate_comprehensive_metrics(y_true, y_pred_proba, y_pred_binary):
    """Calculate comprehensive evaluation metrics"""
    metrics = {}
    
    # AUC-ROC
    metrics['AUC_ROC'] = roc_auc_score(y_true, y_pred_proba)
    
    # AUC-PR (Average Precision)
    metrics['AUC_PR'] = average_precision_score(y_true, y_pred_proba)
    
    # Accuracy
    metrics['Accuracy'] = accuracy_score(y_true, y_pred_binary)
    
    # F1 scores
    metrics['F1_Binary'] = f1_score(y_true, y_pred_binary, average='binary', zero_division=0)
    metrics['F1_Weighted'] = f1_score(y_true, y_pred_binary, average='weighted')
    metrics['F1_Macro'] = f1_score(y_true, y_pred_binary, average='macro')
    
    # Precision and Recall
    metrics['Precision_Binary'] = precision_score(y_true, y_pred_binary, average='binary', zero_division=0)
    metrics['Recall_Binary'] = recall_score(y_true, y_pred_binary, average='binary', zero_division=0)
    metrics['Precision'] = precision_score(y_true, y_pred_binary, average='weighted')
    metrics['Recall'] = recall_score(y_true, y_pred_binary, average='weighted')
    
    return metrics

def load_and_prepare_data():
    """Load and prepare multi-omics data for early integration CNN"""
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
    
    # Early Integration: Concatenate mutation and CNV data
    print("Performing early integration...")
    integrated_data = np.concatenate([mutation_matrix, cnv_matrix], axis=1)
    
    print(f"Integrated data shape: {integrated_data.shape}")
    print(f"Number of genes: {len(gene_names)}")
    print(f"Label distribution: {np.bincount(labels)}")
    
    return integrated_data, labels, gene_names, patient_ids

def train_early_integration_cnn(integrated_data, labels, patient_ids, n_folds=5, n_runs=1,
                                output_dir="early_integration_cnn_results"):
    """Train Early Integration CNN with cross-validation"""
    print("Training Early Integration CNN...")
    
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

    # Device setup (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Setup
    seed = 42
    setup_seed(seed)
    
    # Data preprocessing
    scaler = StandardScaler()
    integrated_data_scaled = scaler.fit_transform(integrated_data)
    
    print(f'Input shape: {integrated_data_scaled.shape}')
    print(f'Labels shape: {labels.shape}')
    
    # Initialize metric storage
    all_metrics = {
        'AUC_ROC': [], 'AUC_PR': [], 'Accuracy': [], 
        'F1_Binary': [], 'F1_Weighted': [], 'F1_Macro': [],
        'Precision_Binary': [], 'Recall_Binary': [], 'Precision': [], 'Recall': []
    }
    all_fold_predictions = []
    run_summary_txt = os.path.join(output_dir, "early_integration_cnn_runwise_summary.txt")
    with open(run_summary_txt, "w") as f:
        f.write("Early Integration CNN - Run-wise Summary (mean ± std across folds)\n")
        f.write(f"n_runs={n_runs}, n_folds={n_folds}\n\n")
    
    for run in range(n_runs):
        print(f"Run {run + 1}/{n_runs}")
        
        # Stratified K-Fold (seed 42 base; offset by run index when n_runs>1)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42 + run)

        
        # Store metrics for this run
        run_metrics = {metric: [] for metric in all_metrics.keys()}
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(integrated_data_scaled, labels)):
            print(f"  Fold {fold + 1}/{n_folds}")
            
            # Split data
            X_train, X_test = integrated_data_scaled[train_idx], integrated_data_scaled[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]
            patient_test = patient_ids[test_idx]
            
            # Stratified split: val = 10% of the 80% train_val pool, giving
            # per-fold ~72% train / ~8% val / ~20% test (5-fold 80/20 outer +
            # 10%-of-train inner val).
            X_train_final, X_val, y_train_final, y_val = train_test_split(
                X_train, y_train,
                test_size=0.10, stratify=y_train, random_state=42 + run,
            )
            
            # Convert to tensors
            y_train_tensor = torch.LongTensor(y_train_final)
            y_val_tensor = torch.LongTensor(y_val)
            y_test_tensor = torch.LongTensor(y_test)
            
            # Create datasets and data loaders
            train_dataset = MultiOmicsDataset(X_train_final, y_train_tensor)
            val_dataset = MultiOmicsDataset(X_val, y_val_tensor)
            test_dataset = MultiOmicsDataset(X_test, y_test_tensor)
            
            pin_memory = (device.type == 'cuda')
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=pin_memory)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
            
            # Create model
            model = EarlyIntegrationCNN(in_dim=integrated_data_scaled.shape[1], num_classes=num_classes)
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(device)
            
            # Calculate class weights for imbalanced data
            class_counts = np.bincount(y_train_final)
            class_weight = torch.tensor(
                [sum(class_counts) / (2 * x) for x in class_counts], dtype=torch.float32
            ).to(device)
            
            # Optimizer and loss function (AdamW + weight decay per SKILLS.md)
            optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            loss_fn = nn.CrossEntropyLoss(weight=class_weight)
            early_stopping = EarlyStopping(patience=patience, verbose=False, delta=delta, stop=stop_epoch)
            
            # Training loop
            for epoch in range(1, epochs + 1):
                # Training phase
                model.train()
                running_loss = 0.0
                y_train_pred = []
                y_train_true = []
                
                for batch_idx, (data, targets) in enumerate(train_loader):
                    batch_idx += 1
                    data = data.to(device, non_blocking=pin_memory)
                    targets = targets.to(device, non_blocking=pin_memory)
                    targets = targets[:, 0] if len(targets.shape) > 1 else targets
                    
                    if batch_idx % grad_acc != 0:
                        outputs = model(data, num_classes=num_classes)
                        loss = loss_fn(outputs, targets)
                        loss.backward()
                    
                    if batch_idx % grad_acc == 0:
                        outputs = model(data, num_classes=num_classes)
                        loss = loss_fn(outputs, targets)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e6))
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    running_loss += loss.item()
                    y_train_pred.extend(outputs[:, 1].detach().cpu().numpy())
                    y_train_true.extend(targets.cpu().numpy())
                
                epoch_loss = running_loss / batch_idx
                ACC_train, AUC_train, f1_weighted_train, f1_macro_train = get_roc(
                    np.array(y_train_true), np.array(y_train_pred))
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                y_val_pred = []
                y_val_true = []
                
                with torch.no_grad():
                    for batch_idx, (data, targets) in enumerate(val_loader):
                        batch_idx += 1
                        data = data.to(device, non_blocking=pin_memory)
                        targets = targets.to(device, non_blocking=pin_memory)
                        targets = targets[:, 0] if len(targets.shape) > 1 else targets
                        outputs = model(data, num_classes=num_classes)
                        loss = loss_fn(outputs, targets)
                        val_loss += loss.item()
                        y_val_pred.extend(outputs[:, 1].detach().cpu().numpy())
                        y_val_true.extend(targets.cpu().numpy())
                
                val_loss = val_loss / batch_idx
                ACC_val, AUC_val, f1_weighted_val, f1_macro_val = get_roc(
                    np.array(y_val_true), np.array(y_val_pred))
                
                # Early stopping check
                early_stopping([f1_macro_val], epoch)
                
                if early_stopping.early_stop and epoch > stop_epoch:
                    print(f"    Early stopping at epoch {epoch}")
                    break
            
            # Test phase
            model.eval()
            y_test_pred = []
            y_test_true = []
            
            with torch.no_grad():
                for data, targets in test_loader:
                    data = data.to(device, non_blocking=pin_memory)
                    targets = targets.to(device, non_blocking=pin_memory)
                    targets = targets[:, 0] if len(targets.shape) > 1 else targets
                    outputs = model(data, num_classes=num_classes)
                    y_test_pred.extend(outputs[:, 1].detach().cpu().numpy())
                    y_test_true.extend(targets.cpu().numpy())
            
            # Calculate test metrics
            y_test_pred = np.array(y_test_pred)
            y_test_pred_binary = (y_test_pred > 0.5).astype(int)

            # Store per-sample predictions (for screenshot-style CSV)
            fold_pred_df = pd.DataFrame({
                'fold': fold + 1,
                'sample': patient_test,
                'y_label': np.array(y_test_true).astype(int),
                'y_score': y_test_pred
            })
            all_fold_predictions.append(fold_pred_df)
            
            fold_metrics = calculate_comprehensive_metrics(y_test_true, y_test_pred, y_test_pred_binary)
            
            # Store fold metrics
            for metric_name, metric_value in fold_metrics.items():
                run_metrics[metric_name].append(metric_value)
            
            print(f"    Fold metrics - AUC: {fold_metrics['AUC_ROC']:.4f}, "
                  f"Accuracy: {fold_metrics['Accuracy']:.4f}, "
                  f"F1-Binary: {fold_metrics['F1_Binary']:.4f}, "
                  f"F1-Weighted: {fold_metrics['F1_Weighted']:.4f}")
        
        # Calculate average metrics for this run (mean ± std across folds)
        run_avg_metrics = {metric: np.mean(values) for metric, values in run_metrics.items()}
        run_std_metrics = {metric: np.std(values, ddof=1) if len(values) > 1 else 0.0
                           for metric, values in run_metrics.items()}
        
        # Store run averages
        for metric_name, avg_value in run_avg_metrics.items():
            all_metrics[metric_name].append(avg_value)
        
        print(f"  Run {run + 1} Metrics (mean ± std over {n_folds} folds):")
        for metric_name in run_avg_metrics.keys():
            print(f"    {metric_name}: {run_avg_metrics[metric_name]:.4f} ± {run_std_metrics[metric_name]:.4f}")
        print()

        with open(run_summary_txt, "a") as f:
            f.write(f"Run {run + 1} (mean ± std over {n_folds} folds)\n")
            for metric_name in run_avg_metrics.keys():
                f.write(
                    f"  {metric_name:16}: {run_avg_metrics[metric_name]:.4f} ± {run_std_metrics[metric_name]:.4f}\n"
                )
            f.write("\n")
    
    # Calculate final statistics
    final_stats = {}
    for metric_name, values in all_metrics.items():
        final_stats[f'{metric_name}_mean'] = np.mean(values)
        final_stats[f'{metric_name}_std'] = np.std(values)
    
    print(f"\n{'='*50}")
    print(f"FINAL RESULTS (Mean ± Std over {n_runs} runs)")
    print(f"{'='*50}")
    for metric in ['AUC_ROC', 'AUC_PR', 'Accuracy', 'F1_Binary', 'F1_Weighted', 'F1_Macro',
                   'Precision_Binary', 'Recall_Binary', 'Precision', 'Recall']:
        mean_val = final_stats[f'{metric}_mean']
        std_val = final_stats[f'{metric}_std']
        print(f"{metric:12}: {mean_val:.4f} ± {std_val:.4f}")
    
    return all_metrics, final_stats, all_fold_predictions

def main():
    """Main function to run Early Integration CNN pipeline"""
    print("Starting Early Integration CNN pipeline...")
    
    # Create output directory
    output_dir = "early_integration_cnn_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load and prepare data
    integrated_data, labels, gene_names, patient_ids = load_and_prepare_data()
    
    # Train model with cross-validation
    all_metrics, final_stats, all_fold_predictions = train_early_integration_cnn(
        integrated_data, labels, patient_ids, output_dir=output_dir
    )
    
    # Save detailed results
    results_df = pd.DataFrame(all_metrics)
    results_df.to_csv(f"{output_dir}/early_integration_cnn_detailed_results.csv", index=False)
    
    # Save summary statistics
    summary_df = pd.DataFrame([final_stats])
    summary_df.to_csv(f"{output_dir}/early_integration_cnn_summary_results.csv", index=False)

    # Save screenshot-style predictions CSV: fold, sample, y_label, y_score
    if all_fold_predictions:
        preds_df = pd.concat(all_fold_predictions, ignore_index=True)
        preds_df = preds_df.sort_values(['fold', 'sample']).reset_index(drop=True)
        preds_df.to_csv(f"{output_dir}/early_integration_cnn_test_predictions.csv", index=False)

    # Save plain-text summary
    summary_txt = f"{output_dir}/early_integration_cnn_summary.txt"
    metric_order = [
        'AUC_ROC', 'AUC_PR', 'Accuracy', 'F1_Binary', 'F1_Weighted', 'F1_Macro',
        'Precision_Binary', 'Recall_Binary', 'Precision', 'Recall'
    ]
    with open(summary_txt, "w") as f:
        f.write("Early Integration CNN - Cross-Validation Summary\n")
        f.write("Metrics reported as mean ± std across runs\n\n")
        for metric in metric_order:
            f.write(
                f"{metric:16}: {final_stats[f'{metric}_mean']:.4f} ± {final_stats[f'{metric}_std']:.4f}\n"
            )
    
    print(f"\nEarly Integration CNN pipeline completed!")
    print(f"Detailed results saved to {output_dir}/early_integration_cnn_detailed_results.csv")
    print(f"Summary results saved to {output_dir}/early_integration_cnn_summary_results.csv")
    print(f"Text summary saved to {summary_txt}")
    print(f"Run-wise text summary saved to {output_dir}/early_integration_cnn_runwise_summary.txt")
    if all_fold_predictions:
        print(f"Predictions CSV saved to {output_dir}/early_integration_cnn_test_predictions.csv")

if __name__ == "__main__":
    main()
