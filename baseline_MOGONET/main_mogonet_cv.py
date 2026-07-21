"""
MOGONET with 5-Fold Cross-Validation
Main script to run the complete CV pipeline
"""
import argparse
import os
import torch
from train_test_cv import train_test_cv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", default=os.environ.get("MOGONET_DATA_FOLDER", "pan_survival_5yr_adapted"),
                        help="Folder containing fold_1..N (output of mogonet_cv_adapter.py).")
    args = parser.parse_args()

    # GPU setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"[MOGONET-CV] Using GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    else:
        print("[MOGONET-CV] Using CPU")

    # Dataset configuration
    data_folder = args.data_folder  # Base folder containing fold_1..N
    view_list = [1, 2]  # 1=CNV, 2=Mutation

    # Training parameters (aligned with SKILLS.md)
    num_epoch_pretrain = 200
    num_epoch = 200
    lr_e_pretrain = 1e-4
    lr_e = 1e-4
    lr_c = 1e-4

    # Binary classification
    num_class = 2

    # Network architecture
    adj_parameter = 10
    dim_he_list = [400, 400, 200]

    # Cross-validation parameters
    n_folds = 5
    patience = 25  # Early stopping patience (in epochs)

    print("="*60)
    print("MOGONET WITH 5-FOLD CROSS-VALIDATION")
    print("="*60)
    print(f"Dataset: {data_folder}")
    print(f"Views: {view_list} (1=CNV, 2=Mutation)")
    print(f"Classes: {num_class} (binary)")
    print(f"Architecture: GCN dims {dim_he_list}")
    print(f"Cross-validation: {n_folds} folds")
    print(f"Early stopping patience: {patience} epochs")
    print(f"Max epochs per fold: {num_epoch}")
    print("="*60)
    
    # Run 5-fold cross-validation
    avg_results, all_results = train_test_cv(
        data_folder, view_list, num_class,
        lr_e_pretrain, lr_e, lr_c, 
        num_epoch_pretrain, num_epoch,
        adj_parameter=adj_parameter,
        dim_he_list=dim_he_list,
        n_folds=n_folds,
        patience=patience
    )
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print("Average Performance Across All Folds:")
    print(f"  Accuracy:  {avg_results['acc']['mean']:.4f} ± {avg_results['acc']['std']:.4f}")
    print(f"  F1 Score:  {avg_results['f1']['mean']:.4f} ± {avg_results['f1']['std']:.4f}")
    print(f"  AUC:       {avg_results['auc']['mean']:.4f} ± {avg_results['auc']['std']:.4f}")
    print(f"  AUPR:      {avg_results['aupr']['mean']:.4f} ± {avg_results['aupr']['std']:.4f}")
    print(f"  Precision: {avg_results['precision']['mean']:.4f} ± {avg_results['precision']['std']:.4f}")
    print(f"  Recall:    {avg_results['recall']['mean']:.4f} ± {avg_results['recall']['std']:.4f}")
    print("="*60)
