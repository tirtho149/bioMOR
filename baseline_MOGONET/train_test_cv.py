""" 
Training and Testing for MOGONET with 5-Fold Cross-Validation
Includes validation set monitoring and early stopping
"""
import os
import csv
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    precision_score, recall_score
)
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from models import init_model_dict, init_optim
from utils_adapted import (
    one_hot_tensor, cal_sample_weight, gen_adj_mat_tensor,
    gen_test_adj_mat_tensor, cal_adj_mat_parameter
)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_amp = False  # disabled: torch.sparse.mm doesn't support FP16 on CUDA
scaler = GradScaler(enabled=use_amp)
print(f"[train_test_cv] Device: {device}")
if device.type == 'cuda':
    try:
        print(f"[train_test_cv] GPU: {torch.cuda.get_device_name(0)}")
    except Exception:
        pass
    torch.backends.cudnn.benchmark = True


def prepare_trte_data(data_folder, view_list):
    """Prepare training, validation, and testing data"""
    num_view = len(view_list)
    labels_tr = np.loadtxt(os.path.join(data_folder, "labels_tr.csv"), delimiter=',').astype(int)
    labels_val = np.loadtxt(os.path.join(data_folder, "labels_val.csv"), delimiter=',').astype(int)
    labels_te = np.loadtxt(os.path.join(data_folder, "labels_te.csv"), delimiter=',').astype(int)

    data_tr_list_np, data_val_list_np, data_te_list_np = [], [], []
    for i in view_list:
        data_tr_list_np.append(np.loadtxt(os.path.join(data_folder, f"{i}_tr.csv"), delimiter=','))
        data_val_list_np.append(np.loadtxt(os.path.join(data_folder, f"{i}_val.csv"), delimiter=','))
        data_te_list_np.append(np.loadtxt(os.path.join(data_folder, f"{i}_te.csv"), delimiter=','))

    num_tr = data_tr_list_np[0].shape[0]
    num_val = data_val_list_np[0].shape[0]
    num_te = data_te_list_np[0].shape[0]

    # Concatenate (train+val+test) per view, then to device
    data_tensor_list = []
    for i in range(num_view):
        mat = np.concatenate((data_tr_list_np[i], data_val_list_np[i], data_te_list_np[i]), axis=0)
        t = torch.tensor(mat, dtype=torch.float32, device=device)
        data_tensor_list.append(t)

    idx_dict = {
        "tr": list(range(num_tr)), 
        "val": list(range(num_tr, num_tr + num_val)),
        "te": list(range(num_tr + num_val, num_tr + num_val + num_te))
    }

    data_train_list, data_all_list = [], []
    for t in data_tensor_list:
        data_train_list.append(t[idx_dict["tr"]].clone())
        data_all_list.append(t.clone())

    labels = np.concatenate((labels_tr, labels_val, labels_te))
    return data_train_list, data_all_list, idx_dict, labels



def gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter, view_list):
    """
    Generate adjacency matrices AND corresponding data subsets for train, val, and test
    RETURNS: (adj_lists, data_lists) for each split
    """
    adj_train_list, adj_val_list, adj_test_list = [], [], []
    data_val_list, data_test_list = [], []  # NEW: return data subsets too
    
    for i in range(len(data_tr_list)):
        view_id = view_list[i]
        adj_metric = "jaccard" if view_id == 2 else "cosine"

        # Compute adj parameter based on training data
        adj_param_adapt = cal_adj_mat_parameter(adj_parameter, data_tr_list[i], adj_metric)

        # Training adjacency (unchanged)
        adj_tr = gen_adj_mat_tensor(data_tr_list[i], adj_param_adapt, adj_metric).to(device)
        
        # VALIDATION: Extract train+val data subset
        tr_idx = trte_idx["tr"]
        val_idx = trte_idx["val"]
        data_tr_val = torch.cat([data_trte_list[i][tr_idx], data_trte_list[i][val_idx]], dim=0)
        
        # Validation adjacency (train+val only, 414 samples)
        adj_val = gen_test_adj_mat_tensor(
            data_trte_list[i], tr_idx, val_idx, adj_param_adapt, adj_metric
        ).to(device)
        
        # TEST: Extract train+test data subset
        te_idx = trte_idx["te"]
        data_tr_te = torch.cat([data_trte_list[i][tr_idx], data_trte_list[i][te_idx]], dim=0)
        
        # Test adjacency (train+test only, 414 samples)
        adj_te = gen_test_adj_mat_tensor(
            data_trte_list[i], tr_idx, te_idx, adj_param_adapt, adj_metric
        ).to(device)

        adj_train_list.append(adj_tr)
        adj_val_list.append(adj_val)
        adj_test_list.append(adj_te)
        data_val_list.append(data_tr_val)
        data_test_list.append(data_tr_te)
        
    return adj_train_list, adj_val_list, adj_test_list, data_val_list, data_test_list


def train_epoch(data_list, adj_list, label, one_hot_label, sample_weight,
                model_dict, optim_dict, train_VCDN=True):
    """Single training epoch with AMP + device placement"""
    loss_dict = {}
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    for m in model_dict:
        model_dict[m].train()

    num_view = len(data_list)

    # Per-view heads
    for i in range(num_view):
        optim = optim_dict[f"C{i+1}"]
        optim.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            emb = model_dict[f"E{i+1}"](data_list[i], adj_list[i])
            logits = model_dict[f"C{i+1}"](emb)
            ci_loss = torch.mean(criterion(logits, label) * sample_weight)

        scaler.scale(ci_loss).backward()
        scaler.step(optim)
        scaler.update()
        loss_dict[f"C{i+1}"] = ci_loss.detach().float().cpu().item()

    # VCDN fusion
    if train_VCDN and num_view >= 2:
        optim_dict["C"].zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            ci_list = []
            for i in range(num_view):
                emb = model_dict[f"E{i+1}"](data_list[i], adj_list[i])
                ci_list.append(model_dict[f"C{i+1}"](emb))
            fused = model_dict["C"](ci_list)
            c_loss = torch.mean(criterion(fused, label) * sample_weight)

        scaler.scale(c_loss).backward()
        scaler.step(optim_dict["C"])
        scaler.update()
        loss_dict["C"] = c_loss.detach().float().cpu().item()

    return loss_dict


@torch.no_grad()
def test_epoch(data_list, adj_list, te_idx, model_dict):
    """Testing/validation epoch with AMP + device placement"""
    for m in model_dict:
        model_dict[m].eval()
    num_view = len(data_list)

    with autocast(enabled=use_amp):
        ci_list = []
        for i in range(num_view):
            emb = model_dict[f"E{i+1}"](data_list[i], adj_list[i])
            ci_list.append(model_dict[f"C{i+1}"](emb))
        c = model_dict["C"](ci_list) if num_view >= 2 else ci_list[0]
        c = c[te_idx, :]
        prob = F.softmax(c, dim=1).float().cpu().numpy()
    return prob


def evaluate_metrics(y_true, y_pred, y_prob):
    """Calculate all evaluation metrics for BINARY classification (positive class = 1)"""
    acc = accuracy_score(y_true, y_pred)
    
    # Binary classification metrics - explicitly set pos_label=1
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.0
    
    try:
        aupr = average_precision_score(y_true, y_prob, pos_label=1)
    except:
        aupr = 0.0
    
    return {
        'acc': acc, 'f1': f1, 'auc': auc, 'aupr': aupr,
        'precision': precision, 'recall': recall
    }


def load_fold_test_samples(data_folder, n_samples):
    """Load test sample IDs for a fold; fallback to generated IDs if unavailable."""
    sample_file = os.path.join(data_folder, "samples_te.csv")
    if os.path.exists(sample_file):
        samples = np.loadtxt(sample_file, dtype=str, delimiter=',')
        samples = np.atleast_1d(samples).tolist()
        if len(samples) == n_samples:
            return samples
    return [f"sample_{i+1}" for i in range(n_samples)]


def save_cv_outputs(data_folder, avg_results, all_results, all_predictions):
    """Save CV summary text and per-sample prediction CSV."""
    summary_path = os.path.join(data_folder, "cv_summary_results.txt")
    pred_path = os.path.join(data_folder, "cv_predictions.csv")
    metrics = ['acc', 'f1', 'auc', 'aupr', 'precision', 'recall']

    with open(summary_path, "w") as f:
        f.write("MOGONET 5-FOLD CROSS-VALIDATION SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write("Average Performance (mean +- std):\n")
        for metric in metrics:
            f.write(
                f"{metric.upper():10s}: "
                f"{avg_results[metric]['mean']:.4f} +- {avg_results[metric]['std']:.4f}\n"
            )
        f.write("\nIndividual Fold Results:\n")
        f.write(f"{'Fold':<6} {'Epoch':<8} {'Acc':<8} {'F1':<8} {'AUC':<8} {'AUPR':<8} {'Prec':<8} {'Recall':<8}\n")
        for r in all_results:
            f.write(
                f"{r['fold']:<6} {r['best_epoch']:<8} {r['acc']:<8.4f} {r['f1']:<8.4f} "
                f"{r['auc']:<8.4f} {r['aupr']:<8.4f} {r['precision']:<8.4f} {r['recall']:<8.4f}\n"
            )

    with open(pred_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "sample", "y_label", "y_score"])
        for row in all_predictions:
            writer.writerow([row["fold"], row["sample"], row["y_label"], f"{row['y_score']:.8f}"])

    return summary_path, pred_path


def train_test_single_fold(data_folder, view_list, num_class,
                           lr_e_pretrain, lr_e, lr_c, 
                           num_epoch_pretrain, num_epoch,
                           adj_parameter=10, dim_he_list=[400, 400, 200],
                           patience=50):
    """
    Train and test on a single fold with validation-based early stopping
    """
    test_interval = 50
    num_view = len(view_list)
    dim_hvcdn = pow(num_class, num_view)

    print(f"Loading data from {data_folder}...")
    data_tr_list, data_trte_list, trte_idx, labels_trte = prepare_trte_data(data_folder, view_list)

    # Labels and weights on device
    labels_tr_tensor = torch.tensor(labels_trte[trte_idx["tr"]], dtype=torch.long, device=device)
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class).to(device)
    sample_weight_tr = torch.tensor(cal_sample_weight(labels_trte[trte_idx["tr"]], num_class),
                                    dtype=torch.float32, device=device)

    print("Generating adjacency matrices...")
    # adj_tr_list, adj_val_list, adj_te_list = gen_trte_adj_mat(
    #     data_tr_list, data_trte_list, trte_idx, adj_parameter, view_list
    # )
    adj_tr_list, adj_val_list, adj_te_list, data_val_list, data_te_list = gen_trte_adj_mat(
    data_tr_list, data_trte_list, trte_idx, adj_parameter, view_list)

    dim_list = [x.shape[1] for x in data_tr_list]
    print(f"Feature dimensions: {dim_list}")
    print(f"GCN hidden dimensions: {dim_he_list}")
    print(f"VCDN dimension: {dim_hvcdn}")

    # Initialize models
    model_dict = init_model_dict(
        num_view, num_class, dim_list, dim_he_list, dim_hvcdn, gcn_dropout=0.2, device=device
    )

    # Pretrain encoders
    print(f"\nPretraining GCNs for {num_epoch_pretrain} epochs...")
    optim_dict = init_optim(num_view, model_dict, lr_e_pretrain, lr_c)

    for epoch in range(num_epoch_pretrain):
        train_epoch(
            data_tr_list, adj_tr_list, labels_tr_tensor,
            onehot_labels_tr_tensor, sample_weight_tr,
            model_dict, optim_dict, train_VCDN=False
        )
        if epoch % 100 == 0:
            print(f"  Pretrain epoch {epoch}")

    # Full training with validation monitoring
    print(f"\nTraining full model for up to {num_epoch} epochs (early stopping patience={patience})...")
    optim_dict = init_optim(num_view, model_dict, lr_e, lr_c)

    best_val_f1 = 0
    best_epoch = 0
    patience_counter = 0
    best_test_results = None
    best_predictions = []

    for epoch in range(num_epoch + 1):
        train_epoch(
            data_tr_list, adj_tr_list, labels_tr_tensor,
            onehot_labels_tr_tensor, sample_weight_tr,
            model_dict, optim_dict, train_VCDN=True
        )

        if epoch % test_interval == 0:
            # Validation evaluation
            # val_prob = test_epoch(data_trte_list, adj_val_list, trte_idx["val"], model_dict)
            # Validation evaluation - use data_val_list instead of data_trte_list
            # Note: indices are now relative to the subset (0 to len(val)-1)
            num_tr = len(trte_idx["tr"])
            val_idx_relative = list(range(num_tr, num_tr + len(trte_idx["val"])))
            val_prob = test_epoch(data_val_list, adj_val_list, val_idx_relative, model_dict)
            y_val_true = labels_trte[trte_idx["val"]]
            y_val_pred = val_prob.argmax(1)
            y_val_prob = val_prob[:, 1] if val_prob.shape[1] > 1 else val_prob[:, 0]
            val_metrics = evaluate_metrics(y_val_true, y_val_pred, y_val_prob)
            
            # Test evaluation
            # te_prob = test_epoch(data_trte_list, adj_te_list, trte_idx["te"], model_dict)
            # Test evaluation - use data_te_list instead of data_trte_list
            # Note: indices are now relative to the subset (0 to len(test)-1)
            te_idx_relative = list(range(num_tr, num_tr + len(trte_idx["te"])))
            te_prob = test_epoch(data_te_list, adj_te_list, te_idx_relative, model_dict)
            y_te_true = labels_trte[trte_idx["te"]]
            y_te_pred = te_prob.argmax(1)
            y_te_prob = te_prob[:, 1] if te_prob.shape[1] > 1 else te_prob[:, 0]
            test_metrics = evaluate_metrics(y_te_true, y_te_pred, y_te_prob)

            print(f"\nEpoch {epoch}:")
            print(f"  Val  - Acc: {val_metrics['acc']:.4f}, F1: {val_metrics['f1']:.4f}, "
                  f"AUC: {val_metrics['auc']:.4f}, AUPR: {val_metrics['aupr']:.4f}")
            print(f"  Test - Acc: {test_metrics['acc']:.4f}, F1: {test_metrics['f1']:.4f}, "
                  f"AUC: {test_metrics['auc']:.4f}, AUPR: {test_metrics['aupr']:.4f}")

            # Early stopping based on validation F1
            # Always capture the first evaluated checkpoint, then require strict improvement.
            if best_test_results is None or val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                best_epoch = epoch
                best_test_results = test_metrics
                te_samples = load_fold_test_samples(data_folder, len(y_te_true))
                best_predictions = [
                    {
                        "sample": te_samples[i],
                        "y_label": int(y_te_true[i]),
                        "y_score": float(y_te_prob[i]),
                    }
                    for i in range(len(y_te_true))
                ]
                patience_counter = 0
                print(f"  → New best validation F1: {best_val_f1:.4f}")
            else:
                patience_counter += test_interval
                
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                print(f"Best validation F1: {best_val_f1:.4f} at epoch {best_epoch}")
                break

    return best_test_results, best_epoch, best_predictions


def train_test_cv(data_folder, view_list, num_class,
                  lr_e_pretrain, lr_e, lr_c, 
                  num_epoch_pretrain, num_epoch,
                  adj_parameter=10, dim_he_list=[400, 400, 200],
                  n_folds=5, patience=50):
    """
    Train and test with 5-fold cross-validation
    """
    print("="*60)
    print("MOGONET 5-FOLD CROSS-VALIDATION")
    print("="*60)
    
    all_results = []
    all_predictions = []
    
    for fold in range(1, n_folds + 1):
        print(f"\n{'='*60}")
        print(f"FOLD {fold}/{n_folds}")
        print(f"{'='*60}")
        
        fold_dir = os.path.join(data_folder, f'fold_{fold}')
        
        if not os.path.exists(fold_dir):
            print(f"Error: Fold directory not found: {fold_dir}")
            continue
        
        fold_results, best_epoch, fold_predictions = train_test_single_fold(
            fold_dir, view_list, num_class,
            lr_e_pretrain, lr_e, lr_c,
            num_epoch_pretrain, num_epoch,
            adj_parameter, dim_he_list, patience
        )
        if fold_results is None:
            print(f"Warning: fold {fold} produced no best results; skipping.")
            continue
        
        fold_results['fold'] = fold
        fold_results['best_epoch'] = best_epoch
        all_results.append(fold_results)
        for row in fold_predictions:
            all_predictions.append({
                "fold": fold,
                "sample": row["sample"],
                "y_label": row["y_label"],
                "y_score": row["y_score"],
            })
        
        print(f"\nFold {fold} Best Results (at epoch {best_epoch}):")
        print(f"  Accuracy:  {fold_results['acc']:.4f}")
        print(f"  F1 Score:  {fold_results['f1']:.4f}")
        print(f"  AUC:       {fold_results['auc']:.4f}")
        print(f"  AUPR:      {fold_results['aupr']:.4f}")
        print(f"  Precision: {fold_results['precision']:.4f}")
        print(f"  Recall:    {fold_results['recall']:.4f}")
    
    # Calculate average results
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    metrics = ['acc', 'f1', 'auc', 'aupr', 'precision', 'recall']
    avg_results = {}
    
    for metric in metrics:
        values = [r[metric] for r in all_results]
        avg = np.mean(values)
        std = np.std(values)
        avg_results[metric] = {'mean': avg, 'std': std}
        
        print(f"{metric.upper():10s}: {avg:.4f} ± {std:.4f}")
    
    # Print individual fold results
    print(f"\n{'='*60}")
    print("INDIVIDUAL FOLD RESULTS")
    print(f"{'='*60}")
    print(f"{'Fold':<6} {'Acc':<8} {'F1':<8} {'AUC':<8} {'AUPR':<8} {'Prec':<8} {'Recall':<8}")
    print("-"*60)
    for r in all_results:
        print(f"{r['fold']:<6} {r['acc']:<8.4f} {r['f1']:<8.4f} {r['auc']:<8.4f} "
              f"{r['aupr']:<8.4f} {r['precision']:<8.4f} {r['recall']:<8.4f}")

    summary_path, pred_path = save_cv_outputs(data_folder, avg_results, all_results, all_predictions)
    print(f"\nSaved summary with std to: {summary_path}")
    print(f"Saved per-sample predictions to: {pred_path}")

    return avg_results, all_results
