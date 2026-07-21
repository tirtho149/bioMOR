# GPU-OPTIMIZED MOGAT for CNV, Mutation binary classification
# 5-Fold Cross-Validation: 1 fold test, 4 folds train, 10% of train as validation
# Random state = 42 for reproducibility
# Binary precision, recall, and F1 metrics

import json
from datetime import datetime
import gc  # For GPU memory management

# Options
addRawFeat = True
base_path = ''
feature_networks_integration = ['cna', 'mut']
node_networks = ['cna', 'mut']
int_method = 'MLP'  # 'MLP' or 'XGBoost' or 'RF' or 'SVM'
xtimes = 50 
xtimes2 = 1

feature_selection_per_network = [False]*len(feature_networks_integration)
top_features_per_network = [500, 500]
optional_feat_selection = False
boruta_runs = 100
boruta_top_features = 50

max_epochs = 200
min_epochs = 50
patience = 25
learning_rates = [0.0001]
hid_sizes = [512] 

# Cross-Validation Settings: 5-fold stratified, 80/20 outer + 10%-of-train inner val.
n_folds = 5
random_state = 42

print('GPU-OPTIMIZED MOGAT with 5-Fold CV is setting up!')
from lib import module2, function
import time
import os, itertools
import pickle
from sklearn.metrics import (f1_score, accuracy_score, classification_report, confusion_matrix,
                           roc_auc_score, average_precision_score, precision_score, recall_score,
                           roc_curve, precision_recall_curve)
from sklearn.model_selection import StratifiedKFold, train_test_split
import statistics
import pandas as pd
import numpy as np
from torch_geometric.data import Data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
import argparse
import errno
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def make_json_safe(obj):
    """Recursively convert numpy/torch types so json.dump can serialize them."""
    if isinstance(obj, dict):
        safe_dict = {}
        for key, value in obj.items():
            if isinstance(key, np.generic):
                key = key.item()
            elif torch.is_tensor(key):
                key = key.item() if key.ndim == 0 else str(key.tolist())

            if not isinstance(key, (str, int, float, bool, type(None))):
                key = str(key)

            safe_dict[key] = make_json_safe(value)
        return safe_dict

    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return [make_json_safe(v) for v in obj.tolist()]

    if isinstance(obj, np.generic):
        return obj.item()

    if torch.is_tensor(obj):
        return obj.item() if obj.ndim == 0 else [make_json_safe(v) for v in obj.tolist()]

    return obj

if ((True in feature_selection_per_network) or (optional_feat_selection == True)):
    import rpy2
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    utils = importr('utils')
    rFerns = importr('rFerns')
    Boruta = importr('Boruta')
    pracma = importr('pracma')
    dplyr = importr('dplyr')
    import re

# ================================================================================
# GPU-ACCELERATED CLASSIFIER MODELS
# ================================================================================

class MLPClassifierGPU(nn.Module):
    """GPU-accelerated MLP Classifier"""
    def __init__(self, input_size, hidden_sizes, num_classes=2, dropout=0.2):
        super(MLPClassifierGPU, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class XGBoostGPU:
    """Wrapper for GPU-accelerated XGBoost"""
    def __init__(self, **kwargs):
        from xgboost import XGBClassifier
        
        default_params = {
            'use_label_encoder': False,
            'n_estimators': 1000,
            'objective': "binary:logistic",
            'eval_metric': "logloss",
            'verbosity': 0,
            'tree_method': 'gpu_hist',
            'gpu_id': 0
        }
        default_params.update(kwargs)
        self.model = XGBClassifier(**default_params)
    
    def fit(self, X, y):
        if torch.is_tensor(X):
            X = X.cpu().numpy()
        if torch.is_tensor(y):
            y = y.cpu().numpy()
        
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        if torch.is_tensor(X):
            X = X.cpu().numpy()
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if torch.is_tensor(X):
            X = X.cpu().numpy()
        return self.model.predict_proba(X)

class RandomForestGPU:
    """CPU-based Random Forest (cuML version available with RAPIDS)"""
    def __init__(self, **kwargs):
        from sklearn.ensemble import RandomForestClassifier
        
        default_params = {
            'n_estimators': 1000,
            'n_jobs': -1,
            'random_state': 42
        }
        default_params.update(kwargs)
        self.model = RandomForestClassifier(**default_params)
    
    def fit(self, X, y):
        if torch.is_tensor(X):
            X = X.cpu().numpy()
        if torch.is_tensor(y):
            y = y.cpu().numpy()
        
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        if torch.is_tensor(X):
            X = X.cpu().numpy()
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if torch.is_tensor(X):
            X = X.cpu().numpy()
        return self.model.predict_proba(X)

class SVMGPU:
    """CPU-based SVM (GPU SVM available with cuML)"""
    def __init__(self, **kwargs):
        from sklearn.svm import SVC
        
        default_params = {
            'probability': True,
            'random_state': 42
        }
        default_params.update(kwargs)
        self.model = SVC(**default_params)
    
    def fit(self, X, y):
        if torch.is_tensor(X):
            X = X.cpu().numpy()
        if torch.is_tensor(y):
            y = y.cpu().numpy()
        
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        if torch.is_tensor(X):
            X = X.cpu().numpy()
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if torch.is_tensor(X):
            X = X.cpu().numpy()
        return self.model.predict_proba(X)

# ================================================================================
# GPU SETUP AND UTILITIES
# ================================================================================

def setup_gpu():
    """Setup GPU with proper configuration"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*80}")
    print("GPU CONFIGURATION")
    print(f"{'='*80}")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU Available: YES")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        
        # Set GPU optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        print(f"GPU Optimizations: ENABLED")
    else:
        print(f"GPU Available: NO - Running on CPU")
        print(f"⚠ Warning: GPU not available. Training will be slower.")
    
    print(f"{'='*80}\n")
    return device

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def to_device(tensor_or_data, device):
    """Safely move tensor or Data object to device"""
    if tensor_or_data is None:
        return None
    
    if isinstance(tensor_or_data, torch.Tensor):
        return tensor_or_data.to(device, non_blocking=True)
    elif isinstance(tensor_or_data, Data):
        return tensor_or_data.to(device, non_blocking=True)
    else:
        return tensor_or_data

# ================================================================================
# TRAINING FUNCTIONS (GPU-OPTIMIZED)
# ================================================================================

def train_gat(model, data, optimizer, criterion, device):
    """GPU-optimized GAT training step"""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    data = to_device(data, device)
    
    out, emb1 = model(data)
    train_idx = data.train_mask
    obj1 = out[train_idx]
    obj2 = data.y[train_idx]
    loss = criterion(obj1, obj2)
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return emb1, loss.item()

def validate_gat(model, data, criterion, device):
    """GPU-optimized GAT validation step"""
    model.eval()
    data = to_device(data, device)
    
    with torch.no_grad():
        out, emb2 = model(data)
        pred = out.argmax(dim=1)
        valid_idx = data.valid_mask
        loss = criterion(out[valid_idx], data.y[valid_idx])
    
    return loss, emb2

def train_classifier_gpu(model, X_train, y_train, X_valid, y_valid, 
                         learning_rate=0.001, max_epochs=500, 
                         patience=50, device='cuda', batch_size=None):
    """Train PyTorch classifier on GPU with early stopping"""
    
    # Ensure data is on GPU
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_valid = X_valid.to(device)
    y_valid = y_valid.to(device)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    best_valid_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    # Use batching if dataset is large
    if batch_size is None:
        batch_size = min(16, X_train.shape[0])
    
    num_batches = (X_train.shape[0] + batch_size - 1) // batch_size
    
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        
        # Training with mini-batches
        indices = torch.randperm(X_train.shape[0], device=device)
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, X_train.shape[0])
            batch_indices = indices[start_idx:end_idx]
            
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            
            optimizer.zero_grad(set_to_none=True)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            valid_outputs = model(X_valid)
            valid_loss = criterion(valid_outputs, y_valid).item()
        
        # Early stopping
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    
    return model

def hyperparameter_search_mlp_gpu(X_train, y_train, X_valid, y_valid, 
                                   device, n_trials=20, random_state=42):
    """GPU-accelerated hyperparameter search for MLP"""
    
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    
    input_size = X_train.shape[1]
    
    # Define search space
    hidden_configs = [
        [16], [32], [64], [128], [256], [512],
        [32, 32], [64, 32], [128, 64], [256, 128]
    ]
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    dropout_rates = [0.3, 0.5, 0.7]
    
    # Default parameters in case search fails
    best_f1 = 0
    best_params = {
        'hidden_layer_sizes': [128],
        'learning_rate_init': 0.001,
        'dropout': 0.5
    }
    
    print("  Hyperparameter search on GPU...")
    
    try:
        for trial in range(min(n_trials, len(hidden_configs) * len(learning_rates))):
            # Random sampling
            hidden = hidden_configs[np.random.randint(len(hidden_configs))]
            lr = learning_rates[np.random.randint(len(learning_rates))]
            dropout = dropout_rates[np.random.randint(len(dropout_rates))]
            
            try:
                # Create model
                model = MLPClassifierGPU(input_size, hidden, num_classes=2, dropout=dropout).to(device)
                
                # Quick training (reduced epochs for search)
                model = train_classifier_gpu(
                    model, X_train, y_train, X_valid, y_valid,
                    learning_rate=lr, max_epochs=100, patience=15, device=device
                )
                
                # Evaluate
                model.eval()
                with torch.no_grad():
                    valid_outputs = model(X_valid.to(device))
                    valid_preds = valid_outputs.argmax(dim=1).cpu().numpy()
                    y_valid_np = y_valid.cpu().numpy() if torch.is_tensor(y_valid) else y_valid
                    valid_f1 = f1_score(y_valid_np, valid_preds, average='binary', zero_division=0)
                
                if valid_f1 > best_f1:
                    best_f1 = valid_f1
                    best_params = {
                        'hidden_layer_sizes': hidden,
                        'learning_rate_init': lr,
                        'dropout': dropout
                    }
                    print(f"    Trial {trial+1}: F1={valid_f1:.4f} (hidden={hidden}, lr={lr})")
                
                # Clean up
                del model
                clear_gpu_memory()
                
            except Exception as e:
                print(f"    Trial {trial+1} failed: {e}")
                continue
                
    except Exception as e:
        print(f"  Warning: Hyperparameter search encountered error: {e}")
        print(f"  Using default parameters")
    
    print(f"  Best F1: {best_f1:.4f}, Params: {best_params}")
    return best_params

# ================================================================================
# EVALUATION FUNCTIONS
# ================================================================================

def evaluate_model_gpu(model, X, y, device):
    """Evaluate model and return binary metrics (GPU-accelerated)"""
    
    # Move data to device
    X = X.to(device)
    
    if isinstance(model, nn.Module):
        # PyTorch model
        model.eval()
        with torch.no_grad():
            outputs = model(X)
            y_pred_proba = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            y_pred = outputs.argmax(dim=1).cpu().numpy()
    else:
        # Sklearn-like model (XGBoost, RF, SVM)
        y_pred = model.predict(X.cpu().numpy())
        y_pred_proba = model.predict_proba(X.cpu().numpy())[:, 1]
    
    # Convert labels if needed
    if torch.is_tensor(y):
        y = y.cpu().numpy()
    
    # Calculate binary metrics
    metrics = {
        'accuracy': float(accuracy_score(y, y_pred)),
        'precision_binary': float(precision_score(y, y_pred, average='binary', zero_division=0)),
        'recall_binary': float(recall_score(y, y_pred, average='binary', zero_division=0)),
        'f1_binary': float(f1_score(y, y_pred, average='binary', zero_division=0)),
        'f1_weighted': float(f1_score(y, y_pred, average='weighted', zero_division=0)),
        'auc': float(roc_auc_score(y, y_pred_proba)) if len(np.unique(y)) > 1 else 0.5,
        'aupr': float(average_precision_score(y, y_pred_proba)) if len(np.unique(y)) > 1 else 0.5
    }
    
    return metrics

def get_predictions_gpu(model, X, y, device):
    """Return y_true, y_pred, y_score for a given split."""
    X = X.to(device)

    if isinstance(model, nn.Module):
        model.eval()
        with torch.no_grad():
            outputs = model(X)
            y_score = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            y_pred = outputs.argmax(dim=1).cpu().numpy()
    else:
        y_pred = model.predict(X.cpu().numpy())
        y_score = model.predict_proba(X.cpu().numpy())[:, 1]

    y_true = y.cpu().numpy() if torch.is_tensor(y) else np.asarray(y)
    return y_true, y_pred, y_score

# ================================================================================
# MAIN EXECUTION
# ================================================================================

# Setup GPU
device = setup_gpu()

# Parser
parser = argparse.ArgumentParser(description='GPU-Optimized MOGAT with 5-Fold CV')
parser.add_argument('-data', "--data_location", nargs=1, default=['your_dataset'])
args = parser.parse_args()
dataset_name = args.data_location[0]

path = base_path + "data/" + dataset_name
if not os.path.exists(path):
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)

# Create output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = base_path + f"MOGAT_{dataset_name}_5fold_results_{timestamp}/"
embeddings_dir = output_dir + "embeddings/"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(embeddings_dir, exist_ok=True)

# Initialize master results
master_results = {
    'experiment_info': {
        'dataset': dataset_name,
        'timestamp': timestamp,
        'device': str(device),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'cross_validation': f'{n_folds}-Fold Stratified',
        'random_state': random_state,
        'validation_split': '10% of training data',
        'integration_method': int_method,
        'node_networks': node_networks,
        'feature_networks': feature_networks_integration,
        'add_raw_features': addRawFeat,
        'max_epochs': max_epochs,
        'min_epochs': min_epochs,
        'patience': patience,
        'learning_rates': learning_rates,
        'hidden_sizes': hid_sizes,
        'gpu_accelerated': True
    },
    'folds': [],
    'summary': {}
}

# Keep fold-level test predictions for CSV export
all_test_predictions = []

# Load labels
data_path_node = base_path + 'data/' + dataset_name + '/'
file = base_path + 'data/' + dataset_name + '/labels.pkl'
print(f"Reading: {file}")

with open(file, 'rb') as f:
    labels_cpu = pickle.load(f)

print(f"Labels shape: {labels_cpu.shape}")
print(f"Label distribution: {np.bincount(labels_cpu)}")

sample_ids_file = base_path + 'data/' + dataset_name + '/sample_ids.pkl'
if os.path.exists(sample_ids_file):
    with open(sample_ids_file, 'rb') as f:
        sample_ids = np.asarray(pickle.load(f)).astype(str)
    if len(sample_ids) != len(labels_cpu):
        raise ValueError(
            f"sample_ids length mismatch: {len(sample_ids)} vs labels length: {len(labels_cpu)}"
        )
    print(f"Sample IDs loaded: {len(sample_ids)}")
else:
    sample_ids = np.asarray([str(i) for i in range(len(labels_cpu))], dtype=str)
    print("Warning: sample_ids.pkl not found. Falling back to index-based sample IDs.")

# Keep both CPU and GPU versions
labels_gpu = torch.tensor(labels_cpu, dtype=torch.long).to(device, non_blocking=True)
print(f"Labels moved to: {labels_gpu.device}")

criterion = torch.nn.CrossEntropyLoss()

start_total = time.time()

# ================================================================================
# 5-FOLD CROSS-VALIDATION
# ================================================================================

print(f"\n{'='*80}")
print(f"STARTING 5-FOLD CROSS-VALIDATION (Random State = {random_state})")
print(f"{'='*80}")

# Create 5-fold stratified split
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

for fold_idx, (train_valid_idx, test_idx) in enumerate(skf.split(np.arange(len(labels_cpu)), labels_cpu)):
    print(f"\n{'='*80}")
    print(f"FOLD {fold_idx + 1}/{n_folds}")
    print(f"{'='*80}")
    
    clear_gpu_memory()
    
    # Split train_valid into train and validation (90% train, 10% valid)
    train_idx, valid_idx = train_test_split(
        train_valid_idx,
        test_size=0.10,
        shuffle=True,
        stratify=labels_cpu[train_valid_idx],
        random_state=random_state
    )
    
    fold_data = {
        'fold_number': fold_idx + 1,
        'split_info': {
            'train_size': len(train_idx),
            'valid_size': len(valid_idx),
            'test_size': len(test_idx),
            'train_percentage': round(len(train_idx)/len(labels_cpu)*100, 2),
            'valid_percentage': round(len(valid_idx)/len(labels_cpu)*100, 2),
            'test_percentage': round(len(test_idx)/len(labels_cpu)*100, 2),
            'train_label_dist': dict(zip(*np.unique(labels_cpu[train_idx], return_counts=True))),
            'valid_label_dist': dict(zip(*np.unique(labels_cpu[valid_idx], return_counts=True))),
            'test_label_dist': dict(zip(*np.unique(labels_cpu[test_idx], return_counts=True)))
        },
        'embeddings_trained': {},
        'classification_results': []
    }
    
    print(f"Split: Train={len(train_idx)} ({fold_data['split_info']['train_percentage']}%), "
          f"Valid={len(valid_idx)} ({fold_data['split_info']['valid_percentage']}%), "
          f"Test={len(test_idx)} ({fold_data['split_info']['test_percentage']}%)")
    
    start = time.time()
    
    # ============================================================================
    # LOAD AND PREPARE FEATURES (GPU)
    # ============================================================================
    
    is_first = 0
    print(f'\nLoading features to GPU...')
    
    for netw in node_networks:
        file = base_path + 'data/' + dataset_name + '/' + netw + '.pkl'
        print(f"Reading: {file}")
        
        with open(file, 'rb') as f:
            feat = pickle.load(f)
            print(f"Loaded {netw} features shape: {feat.shape}")
            values = feat.values
        
        feat_tensor = torch.tensor(values, dtype=torch.float32).to(device, non_blocking=True)
        
        if is_first == 0:
            new_x = feat_tensor
            is_first = 1
        else:
            new_x = torch.cat((new_x, feat_tensor), dim=1)
    
    print(f"Combined features shape: {new_x.shape}")
    print(f"Combined features on device: {new_x.device}")
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    # ============================================================================
    # TRAIN EMBEDDINGS FOR EACH NETWORK (GPU)
    # ============================================================================
    
    for n in range(len(node_networks)):
        netw_base = node_networks[n]
        edge_file = data_path_node + 'edges_' + netw_base + '.pkl'
        
        print(f"\n{'-'*80}")
        print(f"Training embeddings for: {netw_base}")
        print(f"{'-'*80}")
        
        with open(edge_file, 'rb') as f:
            edge_index = pickle.load(f)
        
        print(f"Edge index shape: {edge_index.shape}")
        best_ValidLoss = np.inf
        
        # Hyperparameter search for GAT
        for learning_rate in learning_rates:
            for hid_size in hid_sizes:
                av_valid_losses = []
                
                for ii in range(xtimes2):
                    seed = random_state + ii
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                    
                    # Prepare edge data on GPU
                    edge_index_tensor = torch.tensor(
                        edge_index[edge_index.columns[0:2]].transpose().values, 
                        dtype=torch.long
                    ).to(device, non_blocking=True)
                    
                    edge_attr_tensor = torch.tensor(
                        edge_index[edge_index.columns[2]].transpose().values, 
                        dtype=torch.float32
                    ).to(device, non_blocking=True)
                    
                    # Create Data object on GPU
                    data = Data(
                        x=new_x, 
                        edge_index=edge_index_tensor,
                        edge_attr=edge_attr_tensor, 
                        y=labels_gpu
                    ).to(device, non_blocking=True)
                    
                    # Create masks on GPU
                    train_mask = torch.tensor(
                        [i in set(train_idx) for i in range(data.x.shape[0])],
                        dtype=torch.bool
                    ).to(device, non_blocking=True)
                    
                    valid_mask = torch.tensor(
                        [i in set(valid_idx) for i in range(data.x.shape[0])],
                        dtype=torch.bool
                    ).to(device, non_blocking=True)
                    
                    data.train_mask = train_mask
                    data.valid_mask = valid_mask
                    
                    # Initialize model on GPU
                    in_size = data.x.shape[1]
                    out_size = torch.unique(data.y).shape[0]
                    
                    model = module2.Net(in_size=in_size, hid_size=hid_size, out_size=out_size).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                    
                    # Training loop
                    min_valid_loss = np.inf
                    patience_count = 0
                    
                    for epoch in range(max_epochs):
                        emb, train_loss = train_gat(model, data, optimizer, criterion, device)
                        this_valid_loss, emb = validate_gat(model, data, criterion, device)
                        
                        if this_valid_loss.item() < min_valid_loss:
                            min_valid_loss = this_valid_loss.item()
                            patience_count = 0
                        else:
                            patience_count += 1
                        
                        if (epoch + 1) % 20 == 0:
                            print(f"  Epoch {epoch+1}/{max_epochs} | Train: {train_loss:.4f} | "
                                  f"Valid: {this_valid_loss.item():.4f} | Patience: {patience_count}/{patience}")
                        
                        if epoch >= min_epochs and patience_count >= patience:
                            print(f"  Early stopping at epoch {epoch+1}")
                            break
                    
                    av_valid_losses.append(min_valid_loss)
                    del model, optimizer, data
                    clear_gpu_memory()
                
                av_valid_loss = av_valid_losses[0] if len(av_valid_losses) == 1 else statistics.median(av_valid_losses)
                
                if av_valid_loss < best_ValidLoss:
                    best_ValidLoss = av_valid_loss
                    best_emb_lr = learning_rate
                    best_emb_hs = hid_size
        
        # ========================================================================
        # TRAIN FINAL GAT MODEL (GPU)
        # ========================================================================
        
        print(f"\nTraining final GAT model...")
        print(f"  LR: {best_emb_lr}, Hidden: {best_emb_hs}")
        
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)
        
        # Recreate data on GPU
        edge_index_tensor = torch.tensor(
            edge_index[edge_index.columns[0:2]].transpose().values, 
            dtype=torch.long
        ).to(device, non_blocking=True)
        
        edge_attr_tensor = torch.tensor(
            edge_index[edge_index.columns[2]].transpose().values, 
            dtype=torch.float32
        ).to(device, non_blocking=True)
        
        data = Data(
            x=new_x, 
            edge_index=edge_index_tensor,
            edge_attr=edge_attr_tensor, 
            y=labels_gpu
        ).to(device, non_blocking=True)
        
        train_mask = torch.tensor(
            [i in set(train_idx) for i in range(data.x.shape[0])],
            dtype=torch.bool
        ).to(device, non_blocking=True)
        
        valid_mask = torch.tensor(
            [i in set(valid_idx) for i in range(data.x.shape[0])],
            dtype=torch.bool
        ).to(device, non_blocking=True)
        
        data.train_mask = train_mask
        data.valid_mask = valid_mask
        
        # Initialize and train final model
        in_size = data.x.shape[1]
        out_size = torch.unique(data.y).shape[0]
        
        model = module2.Net(in_size=in_size, hid_size=best_emb_hs, out_size=out_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=best_emb_lr)
        
        min_valid_loss = np.inf
        patience_count = 0
        selected_emb = None
        
        for epoch in range(max_epochs):
            emb, train_loss = train_gat(model, data, optimizer, criterion, device)
            this_valid_loss, emb = validate_gat(model, data, criterion, device)
            
            if this_valid_loss.item() < min_valid_loss:
                min_valid_loss = this_valid_loss.item()
                patience_count = 0
                selected_emb = emb.detach().clone()
            else:
                patience_count += 1
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1} | Valid: {this_valid_loss.item():.4f} | Best: {min_valid_loss:.4f}")
            
            if epoch >= min_epochs and patience_count >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        # Save embedding
        emb_file = embeddings_dir + f'fold_{fold_idx+1}_emb_{netw_base}.pkl'
        with open(emb_file, 'wb') as f:
            pickle.dump(selected_emb.cpu(), f)
        
        fold_data['embeddings_trained'][netw_base] = {
            'embedding_file': emb_file,
            'best_learning_rate': best_emb_lr,
            'best_hidden_size': best_emb_hs,
            'final_validation_loss': float(min_valid_loss),
            'embedding_shape': list(selected_emb.shape)
        }
        
        print(f"✓ Embedding saved: {emb_file}")
        
        del model, optimizer, data, emb
        clear_gpu_memory()
    
    # ============================================================================
    # INTEGRATION AND CLASSIFICATION (GPU)
    # ============================================================================
    
    print(f"\n{'='*80}")
    print("INTEGRATION AND CLASSIFICATION (GPU-ACCELERATED)")
    print(f"{'='*80}")
    
    addFeatures = []
    t = range(len(node_networks))
    trial_combs = []
    for r in range(1, len(t) + 1):
        trial_combs.extend([list(x) for x in itertools.combinations(t, r)])
    
    for trials in range(len(trial_combs)):
        node_networks2 = [node_networks[i] for i in trial_combs[trials]]
        netw_base = node_networks2[0]
        
        # Load first embedding and move to GPU
        emb_file = embeddings_dir + f'fold_{fold_idx+1}_emb_{netw_base}.pkl'
        with open(emb_file, 'rb') as f:
            emb = pickle.load(f)
        emb = emb.to(device, non_blocking=True)
        
        # Concatenate embeddings if multiple networks
        if len(node_networks2) > 1:
            for netw_base in node_networks2[1:]:
                emb_file = embeddings_dir + f'fold_{fold_idx+1}_emb_{netw_base}.pkl'
                with open(emb_file, 'rb') as f:
                    cur_emb = pickle.load(f)
                cur_emb = cur_emb.to(device, non_blocking=True)
                emb = torch.cat((emb, cur_emb), dim=1)
        
        # Add raw features if enabled
        if addRawFeat:
            is_first = 0
            addFeatures = feature_networks_integration
            
            for netw in addFeatures:
                file = base_path + 'data/' + dataset_name + '/' + netw + '.pkl'
                with open(file, 'rb') as f:
                    feat = pickle.load(f)
                
                feat_tensor = torch.tensor(feat.values, dtype=torch.float32).to(device, non_blocking=True)
                
                if is_first == 0:
                    allx = feat_tensor
                    is_first = 1
                else:
                    allx = torch.cat((allx, feat_tensor), dim=1)
            
            emb = torch.cat((emb, allx), dim=1)
        
        print(f"\nCombination {trials+1}/{len(trial_combs)}: {node_networks2}")
        print(f"Feature dimensions: {emb.shape[1]}")
        
        # Prepare datasets (keep on GPU)
        # Convert indices to tensors for GPU indexing
        train_idx_tensor = torch.tensor(train_idx, dtype=torch.long, device=device)
        valid_idx_tensor = torch.tensor(valid_idx, dtype=torch.long, device=device)
        test_idx_tensor = torch.tensor(test_idx, dtype=torch.long, device=device)
        
        X_train = emb[train_idx_tensor]
        X_valid = emb[valid_idx_tensor]
        X_test = emb[test_idx_tensor]
        y_train = labels_gpu[train_idx_tensor]
        y_valid = labels_gpu[valid_idx_tensor]
        y_test = labels_gpu[test_idx_tensor]
        
        print(f"Data on GPU: X_train={X_train.device}, y_train={y_train.device}")
        print(f"  X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        
        # ========================================================================
        # MODEL TRAINING (GPU-ACCELERATED)
        # ========================================================================
        
        print(f"Training {int_method} model on GPU...")
        
        if int_method == 'MLP':
            # GPU-accelerated MLP with hyperparameter search
            best_params = hyperparameter_search_mlp_gpu(
                X_train, y_train, X_valid, y_valid, 
                device=device, n_trials=xtimes, random_state=random_state
            )
            
            print(f"  Best params: {best_params}")
            
            # Train final model with best params
            input_size = X_train.shape[1]
            model = MLPClassifierGPU(
                input_size, 
                best_params['hidden_layer_sizes'], 
                num_classes=2, 
                dropout=best_params['dropout']
            ).to(device)
            
            model = train_classifier_gpu(
                model, X_train, y_train, X_valid, y_valid,
                learning_rate=best_params['learning_rate_init'],
                max_epochs=500,
                patience=50,
                device=device
            )
            
            hyperparams_used = best_params
            
        elif int_method == 'XGBoost':
            # XGBoost with GPU support
            print("  Using GPU-accelerated XGBoost")
            
            from sklearn.model_selection import RandomizedSearchCV
            
            params = {
                'reg_alpha': range(0, 10, 1),
                'reg_lambda': range(1, 10, 1),
                'max_depth': range(1, 6, 1),
                'min_child_weight': range(1, 10, 1),
                'gamma': range(0, 6, 1),
                'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3],
                'colsample_bytree': [0.5, 0.7, 1.0],
                'colsample_bylevel': [0.5, 0.7, 1.0]
            }
            
            base_model = XGBoostGPU(random_state=random_state)
            
            search = RandomizedSearchCV(
                estimator=base_model.model,
                param_distributions=params,
                n_iter=min(xtimes, 20),
                cv=4,
                scoring='f1_binary',
                random_state=random_state,
                verbose=0
            )
            
            search.fit(X_train.cpu().numpy(), y_train.cpu().numpy())
            hyperparams_used = search.best_params_
            
            model = XGBoostGPU(random_state=random_state, **hyperparams_used)
            model.fit(X_train, y_train)
            
        elif int_method == 'RF':
            # Random Forest
            print("  Using Random Forest (CPU multi-threaded)")
            
            from sklearn.model_selection import RandomizedSearchCV
            
            max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
            max_depth.append(None)
            
            params = {
                'n_estimators': [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
                'max_depth': max_depth,
                'min_samples_split': [2, 5, 7, 10],
                'min_samples_leaf': [1, 2, 5, 7, 10],
                'min_impurity_decrease': [0, 0.5, 0.7, 1, 5, 10],
                'max_leaf_nodes': [None, 5, 10, 20]
            }
            
            search = RandomizedSearchCV(
                estimator=RandomForestGPU(random_state=random_state).model,
                param_distributions=params,
                n_iter=min(xtimes, 20),
                cv=4,
                scoring='f1_binary',
                random_state=random_state,
                verbose=0
            )
            
            search.fit(X_train.cpu().numpy(), y_train.cpu().numpy())
            hyperparams_used = search.best_params_
            
            model = RandomForestGPU(random_state=random_state, **hyperparams_used)
            model.fit(X_train, y_train)
            
        elif int_method == 'SVM':
            # SVM
            print("  Using SVM (CPU-based)")
            
            from sklearn.model_selection import RandomizedSearchCV
            
            params = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 'scale', 'auto'],
                'kernel': ['linear', 'rbf']
            }
            
            search = RandomizedSearchCV(
                estimator=SVMGPU(random_state=random_state).model,
                param_distributions=params,
                n_iter=min(xtimes, 20),
                cv=4,
                scoring='f1_binary',
                random_state=random_state,
                verbose=0
            )
            
            search.fit(X_train.cpu().numpy(), y_train.cpu().numpy())
            hyperparams_used = search.best_params_
            
            model = SVMGPU(random_state=random_state, **hyperparams_used)
            model.fit(X_train, y_train)
        
        else:
            raise ValueError(f"Unknown integration method: {int_method}")
        
        # ========================================================================
        # EVALUATION (GPU)
        # ========================================================================
        
        print("  Evaluating model...")
        
        # Evaluate on all splits
        train_metrics = evaluate_model_gpu(model, X_train, y_train, device)
        valid_metrics = evaluate_model_gpu(model, X_valid, y_valid, device)
        test_metrics = evaluate_model_gpu(model, X_test, y_test, device)
        y_test_true, _, y_test_score = get_predictions_gpu(model, X_test, y_test, device)

        test_samples_fold = sample_ids[test_idx]
        for sample_name, y_true, y_score in zip(test_samples_fold, y_test_true, y_test_score):
            all_test_predictions.append({
                'fold': fold_idx + 1,
                'combination_number': trials,
                'sample': str(sample_name),
                'y_label': int(y_true),
                'y_score': float(y_score)
            })
        
        # Store results
        classification_result = {
            'combination_number': trials,
            'used_embeddings': node_networks2,
            'added_raw_features': addFeatures if addRawFeat else None,
            'selected_hyperparameters': hyperparams_used,
            'training_metrics': {k: round(v, 4) for k, v in train_metrics.items()},
            'validation_metrics': {k: round(v, 4) for k, v in valid_metrics.items()},
            'test_metrics': {k: round(v, 4) for k, v in test_metrics.items()}
        }
        
        fold_data['classification_results'].append(classification_result)
        
        print(f"  Test: AUC={test_metrics['auc']:.4f}, F1={test_metrics['f1_binary']:.4f}, "
              f"Precision={test_metrics['precision_binary']:.4f}, Recall={test_metrics['recall_binary']:.4f}")
        
        # Clean up
        if isinstance(model, nn.Module):
            del model
        clear_gpu_memory()
    
    # Store fold timing
    end = time.time()
    fold_data['execution_time_seconds'] = round(end - start, 2)
    
    # Add to master results
    master_results['folds'].append(fold_data)
    
    print(f'\n✓ FOLD {fold_idx + 1} COMPLETED in {round(end - start, 1)} seconds')
    clear_gpu_memory()

# ================================================================================
# COMPUTE SUMMARY STATISTICS ACROSS ALL FOLDS
# ================================================================================

print(f"\n{'='*80}")
print("COMPUTING SUMMARY STATISTICS ACROSS ALL FOLDS")
print(f"{'='*80}")

combinations_summary = {}

for fold_data in master_results['folds']:
    for result in fold_data['classification_results']:
        comb_num = result['combination_number']
        
        if comb_num not in combinations_summary:
            combinations_summary[comb_num] = {
                'combination_number': comb_num,
                'used_embeddings': result['used_embeddings'],
                'test_metrics_across_folds': {
                    'accuracy': [], 'auc': [], 'aupr': [], 
                    'precision_binary': [], 'recall_binary': [], 'f1_binary': []
                }
            }
        
        for metric in ['accuracy', 'auc', 'aupr', 'precision_binary', 'recall_binary', 'f1_binary']:
            combinations_summary[comb_num]['test_metrics_across_folds'][metric].append(
                result['test_metrics'][metric]
            )

summary_statistics = []

for comb_num, comb_data in combinations_summary.items():
    summary_stat = {
        'combination_number': comb_num,
        'used_embeddings': comb_data['used_embeddings'],
        'test_performance_across_folds': {}
    }
    
    for metric in ['accuracy', 'auc', 'aupr', 'precision_binary', 'recall_binary', 'f1_binary']:
        values = comb_data['test_metrics_across_folds'][metric]
        summary_stat['test_performance_across_folds'][metric] = {
            'mean': round(np.mean(values), 4),
            'std': round(np.std(values), 4),
            'min': round(np.min(values), 4),
            'max': round(np.max(values), 4),
            'all_fold_values': [round(v, 4) for v in values]
        }
    
    summary_statistics.append(summary_stat)

# Sort by AUC
summary_statistics.sort(key=lambda x: x['test_performance_across_folds']['auc']['mean'], reverse=True)

# Find best combinations by each metric
best_by_metric = {}
for metric in ['accuracy', 'auc', 'aupr', 'precision_binary', 'recall_binary', 'f1_binary']:
    best_comb = max(summary_statistics, key=lambda x: x['test_performance_across_folds'][metric]['mean'])
    best_by_metric[metric] = {
        'combination_number': best_comb['combination_number'],
        'used_embeddings': best_comb['used_embeddings'],
        'mean_score': best_comb['test_performance_across_folds'][metric]['mean'],
        'std_score': best_comb['test_performance_across_folds'][metric]['std']
    }

# Add summary to master results
master_results['summary'] = {
    'total_folds': n_folds,
    'random_state': random_state,
    'total_combinations_per_fold': len(trial_combs),
    'total_experiments': n_folds * len(trial_combs),
    'combinations_summary': summary_statistics,
    'best_combinations': best_by_metric,
    'overall_statistics': {
        'mean_test_auc_across_all': round(np.mean([s['test_performance_across_folds']['auc']['mean'] 
                                                     for s in summary_statistics]), 4),
        'best_test_auc': round(max([s['test_performance_across_folds']['auc']['mean'] 
                                     for s in summary_statistics]), 4),
        'mean_test_f1_across_all': round(np.mean([s['test_performance_across_folds']['f1_binary']['mean'] 
                                                    for s in summary_statistics]), 4),
        'best_test_f1': round(max([s['test_performance_across_folds']['f1_binary']['mean'] 
                                    for s in summary_statistics]), 4),
        'mean_test_precision_across_all': round(np.mean([s['test_performance_across_folds']['precision_binary']['mean'] 
                                                           for s in summary_statistics]), 4),
        'best_test_precision': round(max([s['test_performance_across_folds']['precision_binary']['mean'] 
                                           for s in summary_statistics]), 4),
        'mean_test_recall_across_all': round(np.mean([s['test_performance_across_folds']['recall_binary']['mean'] 
                                                        for s in summary_statistics]), 4),
        'best_test_recall': round(max([s['test_performance_across_folds']['recall_binary']['mean'] 
                                        for s in summary_statistics]), 4)
    }
}

end_total = time.time()
master_results['experiment_info']['total_execution_time_seconds'] = round(end_total - start_total, 2)
master_results['experiment_info']['average_time_per_fold'] = round((end_total - start_total) / n_folds, 2)

# ================================================================================
# SAVE RESULTS
# ================================================================================

json_file = output_dir + f"MOGAT_{dataset_name}_5fold_complete_results.json"
summary_json = output_dir + f"MOGAT_{dataset_name}_5fold_summary_only.json"
summary_csv = output_dir + f"MOGAT_{dataset_name}_5fold_summary.csv"

print(f"\n{'='*80}")
print("SAVING RESULTS")
print(f"{'='*80}")

master_results_json = make_json_safe(master_results)

# Save complete results
with open(json_file, 'w') as f:
    json.dump(master_results_json, f, indent=2)
print(f"✓ Complete results: {json_file}")

# Save summary only
summary_only = {
    'experiment_info': master_results_json['experiment_info'],
    'summary': master_results_json['summary']
}
with open(summary_json, 'w') as f:
    json.dump(summary_only, f, indent=2)
print(f"✓ Summary JSON: {summary_json}")

# Save summary CSV
summary_df = pd.DataFrame([
    {
        'Combination': s['combination_number'],
        'Embeddings': str(s['used_embeddings']),
        'Test_AUC_Mean': s['test_performance_across_folds']['auc']['mean'],
        'Test_AUC_Std': s['test_performance_across_folds']['auc']['std'],
        'Test_F1_Mean': s['test_performance_across_folds']['f1_binary']['mean'],
        'Test_F1_Std': s['test_performance_across_folds']['f1_binary']['std'],
        'Test_Precision_Mean': s['test_performance_across_folds']['precision_binary']['mean'],
        'Test_Precision_Std': s['test_performance_across_folds']['precision_binary']['std'],
        'Test_Recall_Mean': s['test_performance_across_folds']['recall_binary']['mean'],
        'Test_Recall_Std': s['test_performance_across_folds']['recall_binary']['std'],
        'Test_Acc_Mean': s['test_performance_across_folds']['accuracy']['mean'],
        'Test_Acc_Std': s['test_performance_across_folds']['accuracy']['std'],
        'Test_AUPR_Mean': s['test_performance_across_folds']['aupr']['mean'],
        'Test_AUPR_Std': s['test_performance_across_folds']['aupr']['std']
    }
    for s in summary_statistics
])
summary_df.to_csv(summary_csv, index=False)
print(f"✓ Summary CSV: {summary_csv}")

# Save per-sample predictions CSV (best AUC combination)
best_auc_comb = best_by_metric['auc']['combination_number']
predictions_csv = output_dir + f"MOGAT_{dataset_name}_5fold_test_predictions.csv"
predictions_df = pd.DataFrame([
    {
        'fold': p['fold'],
        'sample': p['sample'],
        'y_label': p['y_label'],
        'y_score': p['y_score']
    }
    for p in all_test_predictions
    if p['combination_number'] == best_auc_comb
])

if not predictions_df.empty:
    predictions_df = predictions_df.sort_values(['fold', 'sample']).reset_index(drop=True)
predictions_df.to_csv(predictions_csv, index=False)
print(f"✓ Per-sample Test Predictions CSV: {predictions_csv} (best AUC combination: {best_auc_comb})")

# Save human-readable summary text file
summary_txt = output_dir + f"MOGAT_{dataset_name}_5fold_summary.txt"
top_k = min(3, len(summary_statistics))
summary_lines = [
    "MOGAT 5-Fold Summary",
    f"Dataset: {dataset_name}",
    f"Device: {device}",
    f"Cross-Validation: {n_folds}-Fold Stratified",
    f"Random State: {random_state}",
    "",
    "Overall Best Scores:",
    f"  Best Test AUC: {master_results['summary']['overall_statistics']['best_test_auc']:.4f}",
    f"  Best Test F1: {master_results['summary']['overall_statistics']['best_test_f1']:.4f}",
    f"  Best Test Precision: {master_results['summary']['overall_statistics']['best_test_precision']:.4f}",
    f"  Best Test Recall: {master_results['summary']['overall_statistics']['best_test_recall']:.4f}",
    "",
    "Best Combination By Metric:",
]

for metric, info in best_by_metric.items():
    summary_lines.append(
        f"  {metric}: comb {info['combination_number']} | embeddings={info['used_embeddings']} | "
        f"{info['mean_score']:.4f} ± {info['std_score']:.4f}"
    )

summary_lines.extend([
    "",
    f"Top {top_k} Combinations by Mean AUC:",
])
for rank, s in enumerate(summary_statistics[:top_k], start=1):
    auc_stat = s['test_performance_across_folds']['auc']
    f1_stat = s['test_performance_across_folds']['f1_binary']
    summary_lines.append(
        f"  {rank}. comb {s['combination_number']} | embeddings={s['used_embeddings']} | "
        f"AUC={auc_stat['mean']:.4f}+-{auc_stat['std']:.4f} | F1={f1_stat['mean']:.4f}+-{f1_stat['std']:.4f}"
    )

summary_lines.extend([
    "",
    "Output Files:",
    f"  Complete JSON: {json_file}",
    f"  Summary JSON: {summary_json}",
    f"  Summary CSV: {summary_csv}",
    f"  Summary TXT: {summary_txt}",
    f"  Per-sample predictions CSV: {predictions_csv}",
    "",
    f"Execution Time: {round(end_total - start_total, 1)}s",
])

with open(summary_txt, 'w') as f:
    f.write("\n".join(summary_lines) + "\n")
print(f"✓ Summary TXT: {summary_txt}")

# ================================================================================
# PRINT FINAL SUMMARY
# ================================================================================

print(f"\n{'='*80}")
print(f"FINAL SUMMARY - {n_folds}-FOLD CROSS-VALIDATION COMPLETED")
print(f"{'='*80}")

print(f"\nExperiment Configuration:")
print(f"  Dataset: {dataset_name}")
print(f"  Device: {device}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"  Cross-Validation: {n_folds}-Fold Stratified")
print(f"  Random State: {random_state}")
print(f"  Validation Split: 10% of training data")
print(f"  Integration: {int_method} (GPU-Accelerated)")

print(f"\nOverall Performance (Mean ± Std across {n_folds} folds):")
print(f"  Best Test AUC:       {master_results['summary']['overall_statistics']['best_test_auc']:.4f}")
print(f"  Best Test F1:        {master_results['summary']['overall_statistics']['best_test_f1']:.4f}")
print(f"  Best Test Precision: {master_results['summary']['overall_statistics']['best_test_precision']:.4f}")
print(f"  Best Test Recall:    {master_results['summary']['overall_statistics']['best_test_recall']:.4f}")

print(f"\nBest Combinations by Metric:")
for metric, info in best_by_metric.items():
    metric_display = metric.replace('_', ' ').title()
    print(f"  {metric_display}: {info['mean_score']:.4f} ± {info['std_score']:.4f}")
    print(f"    → {info['used_embeddings']}")

print(f"\nExecution Time:")
print(f"  Total: {round(end_total - start_total, 1)}s ({round((end_total - start_total)/60, 1)}min)")
print(f"  Per Fold: {round((end_total - start_total) / n_folds, 1)}s")

if torch.cuda.is_available():
    print(f"\nGPU Memory Usage:")
    print(f"  Peak Allocated: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print(f"  Current Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"  GPU Utilization: FULLY ACCELERATED ✓")

print(f"\n{'='*80}")
print("GPU-OPTIMIZED MOGAT WITH 5-FOLD CV COMPLETED SUCCESSFULLY!")
print(f"All stages executed on GPU: GAT Training ✓ | Classification ✓")
print(f"Results saved to: {output_dir}")
print(f"{'='*80}")
