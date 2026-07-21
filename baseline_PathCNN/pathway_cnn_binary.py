import pandas as pd
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from keras.callbacks import EarlyStopping
from keras import backend as K
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (roc_auc_score, confusion_matrix, accuracy_score,
                           precision_recall_curve, auc, f1_score, 
                           precision_score, recall_score, classification_report)
from sklearn.decomposition import PCA
from keras.optimizers import Adam
import warnings
import os
import json
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """
    Load and preprocess mutation, CNV, labels, and pathway data
    Ensure both mutation and CNV data have the same genes in the same order
    """
    print("Loading data...")
    
    # Load data files
    mutation_data = pd.read_csv("mutation_data.csv")
    cnv_data = pd.read_csv("cnv_data.csv")
    labels_data = pd.read_csv("labels.csv")
    pathway_genes = pd.read_csv("pathway_genes.csv")
    
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
    
    # Get gene names from both datasets
    mutation_genes = set(mutation_aligned.columns[1:].tolist())
    cnv_genes = set(cnv_aligned.columns[1:].tolist())
    
    # Find common genes between mutation and CNV data
    common_genes = sorted(list(mutation_genes & cnv_genes))
    print(f"Found {len(common_genes)} common genes between mutation and CNV data")
    
    # Filter both datasets to only include common genes
    mutation_aligned = mutation_aligned[['patient_id'] + common_genes] if 'patient_id' in mutation_aligned.columns else mutation_aligned[[mutation_aligned.columns[0]] + common_genes]
    cnv_aligned = cnv_aligned[['patient_id'] + common_genes] if 'patient_id' in cnv_aligned.columns else cnv_aligned[[cnv_aligned.columns[0]] + common_genes]
    
    # Extract gene names and data matrices (now they match)
    gene_names = common_genes
    mutation_matrix = mutation_aligned.iloc[:, 1:].values
    cnv_matrix = cnv_aligned.iloc[:, 1:].values
    labels = labels_aligned.iloc[:, 1].values.astype(int)  # <-- FIX: Convert to int
    
    print(f"Mutation data shape: {mutation_matrix.shape}")
    print(f"CNV data shape: {cnv_matrix.shape}")
    print(f"Number of common genes: {len(gene_names)}")
    print(f"Label distribution: {np.bincount(labels)}")
    
    # Verify shapes match
    assert mutation_matrix.shape[1] == cnv_matrix.shape[1], "Mutation and CNV matrices must have same number of genes"
    assert len(gene_names) == mutation_matrix.shape[1], "Gene names length must match data matrix columns"
    
    return mutation_matrix, cnv_matrix, labels, gene_names, pathway_genes

# def load_and_preprocess_data():
#     """
#     Load and preprocess mutation, CNV, labels, and pathway data
#     Ensure both mutation and CNV data have the same genes in the same order
#     """
#     print("Loading data...")
    
#     # Load data files
#     mutation_data = pd.read_csv("mutation_data.csv")
#     cnv_data = pd.read_csv("cnv_data.csv")
#     labels_data = pd.read_csv("labels.csv")
#     pathway_genes = pd.read_csv("pathway_genes.csv")
    
#     # Extract patient IDs and align datasets
#     mutation_patients = mutation_data.iloc[:, 0].values
#     cnv_patients = cnv_data.iloc[:, 0].values
#     label_patients = labels_data.iloc[:, 0].values
    
#     # Find common patients across all datasets
#     common_patients = list(set(mutation_patients) & set(cnv_patients) & set(label_patients))
#     print(f"Found {len(common_patients)} common patients across all datasets")
    
#     # Align datasets by common patients
#     mutation_aligned = mutation_data[mutation_data.iloc[:, 0].isin(common_patients)].reset_index(drop=True)
#     cnv_aligned = cnv_data[cnv_data.iloc[:, 0].isin(common_patients)].reset_index(drop=True)
#     labels_aligned = labels_data[labels_data.iloc[:, 0].isin(common_patients)].reset_index(drop=True)
    
#     # Sort by patient ID to ensure alignment
#     mutation_aligned = mutation_aligned.sort_values(by=mutation_aligned.columns[0]).reset_index(drop=True)
#     cnv_aligned = cnv_aligned.sort_values(by=cnv_aligned.columns[0]).reset_index(drop=True)
#     labels_aligned = labels_aligned.sort_values(by=labels_aligned.columns[0]).reset_index(drop=True)
    
#     # Get gene names from both datasets
#     mutation_genes = set(mutation_aligned.columns[1:].tolist())
#     cnv_genes = set(cnv_aligned.columns[1:].tolist())
    
#     # Find common genes between mutation and CNV data
#     common_genes = sorted(list(mutation_genes & cnv_genes))
#     print(f"Found {len(common_genes)} common genes between mutation and CNV data")
    
#     # Filter both datasets to only include common genes
#     mutation_aligned = mutation_aligned[['patient_id'] + common_genes] if 'patient_id' in mutation_aligned.columns else mutation_aligned[[mutation_aligned.columns[0]] + common_genes]
#     cnv_aligned = cnv_aligned[['patient_id'] + common_genes] if 'patient_id' in cnv_aligned.columns else cnv_aligned[[cnv_aligned.columns[0]] + common_genes]
    
#     # Extract gene names and data matrices (now they match)
#     gene_names = common_genes
#     mutation_matrix = mutation_aligned.iloc[:, 1:].values
#     cnv_matrix = cnv_aligned.iloc[:, 1:].values
#     labels = labels_aligned.iloc[:, 1].values
    
#     print(f"Mutation data shape: {mutation_matrix.shape}")
#     print(f"CNV data shape: {cnv_matrix.shape}")
#     print(f"Number of common genes: {len(gene_names)}")
#     print(f"Label distribution: {np.bincount(labels)}")
    
#     # Verify shapes match
#     assert mutation_matrix.shape[1] == cnv_matrix.shape[1], "Mutation and CNV matrices must have same number of genes"
#     assert len(gene_names) == mutation_matrix.shape[1], "Gene names length must match data matrix columns"
    
#     return mutation_matrix, cnv_matrix, labels, gene_names, pathway_genes

def create_pathway_gene_mapping(pathway_genes_df, available_genes):
    """
    Create pathway to gene mapping and filter available genes
    """
    print("Creating pathway-gene mapping...")
    
    pathway_mapping = {}
    pathway_names = {}
    
    for _, row in pathway_genes_df.iterrows():
        pathway_id = row['Pathway_ID']
        pathway_name = row['Pathway_Name']
        genes = str(row['Genes']).split(',')  # Genes are separated by comma
        
        # Filter genes that exist in our dataset
        valid_genes = [gene.strip() for gene in genes if gene.strip() in available_genes]
        
        if len(valid_genes) >= 5:  # Only include pathways with at least 5 genes
            pathway_mapping[pathway_id] = valid_genes
            pathway_names[pathway_id] = pathway_name
    
    print(f"Found {len(pathway_mapping)} pathways with sufficient genes")
    return pathway_mapping, pathway_names

# def apply_pca_to_pathways(data_matrix, pathway_mapping, gene_names, n_components=5):
#     """
#     Apply PCA to each pathway separately
#     """
#     print(f"Applying PCA to pathways... (data shape: {data_matrix.shape})")
    
#     n_samples = data_matrix.shape[0]
#     n_pathways = len(pathway_mapping)
    
#     # Create gene name to index mapping
#     gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
    
#     print(f"Number of genes in mapping: {len(gene_to_idx)}")
#     print(f"Data matrix has {data_matrix.shape[1]} columns")
    
#     pathway_pca_data = np.zeros((n_samples, n_pathways, n_components))
#     pathway_list = list(pathway_mapping.keys())
    
#     for pathway_idx, pathway_id in enumerate(pathway_list):
#         pathway_genes = pathway_mapping[pathway_id]
        
#         # Get gene indices for this pathway - only include genes that exist in gene_to_idx
#         gene_indices = [gene_to_idx[gene] for gene in pathway_genes if gene in gene_to_idx]
        
#         if len(gene_indices) > 0:
#             # Verify indices are valid
#             max_idx = max(gene_indices)
#             if max_idx >= data_matrix.shape[1]:
#                 print(f"WARNING: Pathway {pathway_id} has invalid gene index {max_idx} (max allowed: {data_matrix.shape[1]-1})")
#                 continue
            
#             # Extract pathway data
#             pathway_data = data_matrix[:, gene_indices]
            
#             # Apply PCA
#             n_comp = min(n_components, len(gene_indices), n_samples - 1)
#             if n_comp > 0:
#                 pca = PCA(n_components=n_comp)
#                 pca_result = pca.fit_transform(pathway_data)
                
#                 # Fill in the PCA results (pad with zeros if fewer components)
#                 pathway_pca_data[:, pathway_idx, :n_comp] = pca_result
    
#     print(f"PCA completed. Result shape: {pathway_pca_data.shape}")
#     return pathway_pca_data, pathway_list

def apply_pca_to_pathways(data_matrix, pathway_mapping, gene_names, n_components=5):
    """
    Apply PCA to each pathway separately with robust error handling
    """
    print(f"Applying PCA to pathways... (data shape: {data_matrix.shape})")
    
    n_samples = data_matrix.shape[0]
    n_pathways = len(pathway_mapping)
    
    # Create gene name to index mapping
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
    
    print(f"Number of genes in mapping: {len(gene_to_idx)}")
    print(f"Data matrix has {data_matrix.shape[1]} columns")
    
    # Check for NaN or inf values in the data
    if np.isnan(data_matrix).any():
        print("WARNING: NaN values found in data matrix. Replacing with 0.")
        data_matrix = np.nan_to_num(data_matrix, nan=0.0)
    
    if np.isinf(data_matrix).any():
        print("WARNING: Inf values found in data matrix. Replacing with 0.")
        data_matrix = np.nan_to_num(data_matrix, posinf=0.0, neginf=0.0)
    
    pathway_pca_data = np.zeros((n_samples, n_pathways, n_components))
    pathway_list = list(pathway_mapping.keys())
    
    failed_pathways = []
    
    for pathway_idx, pathway_id in enumerate(pathway_list):
        pathway_genes = pathway_mapping[pathway_id]
        
        # Get gene indices for this pathway - only include genes that exist in gene_to_idx
        gene_indices = [gene_to_idx[gene] for gene in pathway_genes if gene in gene_to_idx]
        
        if len(gene_indices) > 0:
            # Verify indices are valid
            max_idx = max(gene_indices)
            if max_idx >= data_matrix.shape[1]:
                print(f"WARNING: Pathway {pathway_id} has invalid gene index {max_idx} (max allowed: {data_matrix.shape[1]-1})")
                continue
            
            # Extract pathway data
            pathway_data = data_matrix[:, gene_indices]
            
            # Check for NaN/inf in pathway data
            if np.isnan(pathway_data).any() or np.isinf(pathway_data).any():
                pathway_data = np.nan_to_num(pathway_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Check if pathway has any variance
            if np.std(pathway_data) < 1e-10:
                print(f"WARNING: Pathway {pathway_id} has zero or near-zero variance. Skipping PCA.")
                continue
            
            # Apply PCA with error handling
            n_comp = min(n_components, len(gene_indices), n_samples - 1)
            if n_comp > 0:
                try:
                    # Standardize the data before PCA to improve numerical stability
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    pathway_data_scaled = scaler.fit_transform(pathway_data)
                    
                    # Replace any remaining NaN/inf after scaling
                    pathway_data_scaled = np.nan_to_num(pathway_data_scaled, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # Use randomized SVD solver which is more robust
                    pca = PCA(n_components=n_comp, svd_solver='randomized', random_state=42)
                    pca_result = pca.fit_transform(pathway_data_scaled)
                    
                    # Fill in the PCA results (pad with zeros if fewer components)
                    pathway_pca_data[:, pathway_idx, :n_comp] = pca_result
                    
                except Exception as e:
                    print(f"WARNING: PCA failed for pathway {pathway_id}: {str(e)}")
                    failed_pathways.append(pathway_id)
                    # Leave as zeros (already initialized)
                    continue
    
    if failed_pathways:
        print(f"\nPCA failed for {len(failed_pathways)} pathways: {failed_pathways[:5]}..." if len(failed_pathways) > 5 else failed_pathways)
    
    print(f"PCA completed. Result shape: {pathway_pca_data.shape}")
    return pathway_pca_data, pathway_list

def create_pathway_images(mutation_pca, cnv_pca, n_pc=2):
    """
    Create pathway images by combining PCA results from different omics
    """
    print(f"Creating pathway images with {n_pc} principal components...")
    
    n_samples, n_pathways, _ = mutation_pca.shape
    
    # Use first n_pc components from each omics type
    pathway_images = np.zeros((n_samples, n_pathways, n_pc * 2))  # 2 omics types
    
    for i in range(n_samples):
        mutation_pc = mutation_pca[i, :, :n_pc]
        cnv_pc = cnv_pca[i, :, :n_pc]
        
        # Concatenate along the feature dimension
        pathway_images[i, :, :] = np.concatenate([mutation_pc, cnv_pc], axis=1)
    
    return pathway_images

def order_pathways_by_correlation(pathway_images):
    """
    Order pathways by correlation to cluster similar pathways together
    """
    print("Ordering pathways by correlation...")
    
    n_samples, n_pathways, n_features = pathway_images.shape
    
    # Flatten pathway images for correlation calculation
    flattened_data = pathway_images.reshape(n_pathways, n_samples * n_features)
    
    # Calculate correlation matrix
    correlation_matrix = np.corrcoef(flattened_data)
    
    # Order pathways by correlation
    ordered_indices = [0]  # Start with first pathway
    remaining_indices = list(range(1, n_pathways))
    
    while remaining_indices:
        last_pathway = ordered_indices[-1]
        correlations = [correlation_matrix[last_pathway, idx] for idx in remaining_indices]
        next_pathway_pos = np.argmax(correlations)
        next_pathway = remaining_indices.pop(next_pathway_pos)
        ordered_indices.append(next_pathway)
    
    # Reorder pathway images
    reordered_images = pathway_images[:, ordered_indices, :]
    
    return reordered_images, ordered_indices

def create_pathcnn_model(input_shape, num_classes=2):
    """
    Create PathCNN model with original architecture
    """
    print("Creating PathCNN model...")
    
    # Input layers
    image_input = Input(shape=input_shape)
    other_data_input = Input(shape=(1,))  # For age or other clinical data
    
    # First convolution
    conv1 = Conv2D(32, kernel_size=(3, 3),
                   activation='relu', padding='same')(image_input)
    
    # Second Convolution
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    conv2 = MaxPooling2D(pool_size=(4, 2))(conv2)
    conv2 = Dropout(0.25)(conv2)
    first_part_output = Flatten()(conv2)
    
    # Without clinical data - using only pathway images
    merged_model = first_part_output
    merged_model = Dense(64, activation='relu')(merged_model)
    merged_model = Dropout(0.5)(merged_model)
    
    predictions = Dense(num_classes, activation='softmax')(merged_model)
    
    # Create model
    model = Model(inputs=[image_input, other_data_input], outputs=predictions)
    
    return model

def calculate_metrics(y_true, y_pred_proba, y_pred_binary, pos_label=1):
    """
    Calculate evaluation metrics with BINARY averaging (positive = pos_label).
    """
    metrics = {}

    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)

    # AUC (ROC) – uses probability of the positive class
    metrics['auc'] = roc_auc_score(y_true, y_pred_proba)

    # AUPR
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    metrics['aupr'] = auc(recall, precision)

    # 🔁 Binary metrics (was: average='macro')
    metrics['f1'] = f1_score(y_true, y_pred_binary, average='binary', pos_label=pos_label)
    metrics['precision'] = precision_score(y_true, y_pred_binary, average='binary', pos_label=pos_label)
    metrics['recall'] = recall_score(y_true, y_pred_binary, average='binary', pos_label=pos_label)

    return metrics


def train_pathcnn_multiple_runs(pathway_images, labels, n_folds=5):
    """
    Train PathCNN with 5-fold stratified cross-validation.
    Per fold: ~72% train / ~8% val / ~20% test (80/20 outer + 10%-of-train val).
    """
    print(f"Training PathCNN with {n_folds}-fold stratified CV...")
    
    # Create results directory
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Parameters
    batch_size = 16   # SKILLS.md alignment (was 64)
    min_epochs = 50
    max_epochs = 200
    patience = 25
    num_classes = 2
    
    # Input shape
    img_rows, img_cols = pathway_images.shape[1], pathway_images.shape[2]
    input_shape = (img_rows, img_cols, 1)
    
    # Reshape for CNN input
    X = pathway_images.reshape(pathway_images.shape[0], img_rows, img_cols, 1)
    X = X.astype('float32')
    y = labels
    
    print(f'Input shape: {X.shape}')
    print(f'Labels shape: {y.shape}')
    
    # Initialize storage for all runs
    all_runs_results = []
    all_metrics = {
        'accuracy': [], 'auc': [], 'aupr': [], 'f1': [], 'precision': [], 'recall': []
    }
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    for run, (train_val_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*60}")
        print(f"FOLD {run + 1}/{n_folds}")
        print(f"{'='*60}")

        # Outer 5-fold split: ~80% train_val / ~20% test
        X_train_val, X_test = X[train_val_idx], X[test_idx]
        y_train_val, y_test = y[train_val_idx], y[test_idx]

        # Inner split: val = 10% of train_val (≈ 8% of total)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.10, stratify=y_train_val, random_state=42
        )
        
        print(f"Training set size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
        print(f"Validation set size: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
        print(f"Test set size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
        
        # Create dummy clinical data (zeros)
        dummy_train = np.zeros((len(X_train), 1))
        dummy_val = np.zeros((len(X_val), 1))
        dummy_test = np.zeros((len(X_test), 1))
        
        # Convert labels to categorical
        y_train_cat = keras.utils.to_categorical(y_train, num_classes)
        y_val_cat = keras.utils.to_categorical(y_val, num_classes)
        y_test_cat = keras.utils.to_categorical(y_test, num_classes)
        
        # Create and compile model
        model = create_pathcnn_model(input_shape, num_classes)
        
        # SKILLS.md alignment: AdamW with weight_decay=5e-4 (was Adam).
        # Keras 2.11+ has AdamW built in; fall back to legacy Adam if not.
        try:
            from keras.optimizers import AdamW
            optimizer = AdamW(learning_rate=0.0001, weight_decay=5e-4,
                              beta_1=0.9, beta_2=0.999)
        except ImportError:
            optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
        model.compile(optimizer=optimizer, 
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        
        # Calculate class weights for imbalanced data
        class_counts = np.bincount(y_train)
        class_weight = {0: 1.0, 1: class_counts[0] / class_counts[1]} if len(class_counts) > 1 else {0: 1.0}
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        
        print(f"Training model with early stopping (min_epochs={min_epochs}, patience={patience})...")
        
        # Train model
        history = model.fit([X_train, dummy_train], y_train_cat,
                          batch_size=batch_size,
                          epochs=max_epochs,
                          verbose=1,
                          class_weight=class_weight,
                          validation_data=([X_val, dummy_val], y_val_cat),
                          callbacks=[early_stopping])
        
        # Ensure minimum epochs
        epochs_trained = len(history.history['loss'])
        if epochs_trained < min_epochs:
            print(f"Training additional {min_epochs - epochs_trained} epochs to reach minimum...")
            additional_history = model.fit([X_train, dummy_train], y_train_cat,
                                         batch_size=batch_size,
                                         epochs=min_epochs - epochs_trained,
                                         verbose=1,
                                         class_weight=class_weight,
                                         validation_data=([X_val, dummy_val], y_val_cat))
        
        # Predict on test set
        y_pred = model.predict([X_test, dummy_test], verbose=0)
        
        # Get probability for positive class and binary predictions
        y_pred_proba = y_pred[:, 1]  # Probability of class 1
        y_pred_binary = np.argmax(y_pred, axis=1)
        
        # Calculate all metrics
        run_metrics = calculate_metrics(y_test, y_pred_proba, y_pred_binary)
        
        # Store metrics for this run
        for metric_name, metric_value in run_metrics.items():
            all_metrics[metric_name].append(metric_value)
        
        # Store run information
        run_info = {
            'run': run + 1,
            'random_state': 42,
            'epochs_trained': len(history.history['loss']),
            'metrics': run_metrics,
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'class_distribution_train': np.bincount(y_train).tolist(),
            'class_distribution_test': np.bincount(y_test).tolist()
        }
        all_runs_results.append(run_info)
        
        # Print run results
        print(f"\nRUN {run + 1} RESULTS:")
        print(f"Epochs trained: {epochs_trained}")
        print(f"Accuracy: {run_metrics['accuracy']:.4f}")
        print(f"AUC: {run_metrics['auc']:.4f}")
        print(f"AUPR: {run_metrics['aupr']:.4f}")
        print(f"F1: {run_metrics['f1']:.4f}")
        print(f"Precision: {run_metrics['precision']:.4f}")
        print(f"Recall: {run_metrics['recall']:.4f}")
        
        # Save individual run results
        with open(f'results/run_{run+1}_results.json', 'w') as f:
            json.dump(run_info, f, indent=4)
    
    # Calculate summary statistics - mean ± std
    mean_std = {}
    for metric_name in ['auc', 'aupr', 'accuracy', 'f1', 'precision', 'recall']:
        if all_metrics[metric_name]:
            mean_std[metric_name] = {
                "mean": float(np.mean(all_metrics[metric_name])),
                "std": float(np.std(all_metrics[metric_name]))
            }
    
    print("\n" + "="*60)
    print("FINAL RESULTS (Mean ± Std over runs)")
    print("="*60)
    for metric in ['auc', 'aupr', 'accuracy', 'f1', 'precision', 'recall']:
        if metric in mean_std:
            mu = mean_std[metric]["mean"]
            sd = mean_std[metric]["std"]
            print(f"{metric.upper():12s}: {mu:.4f} ± {sd:.4f}")
    
    # Save mean ± std results
    pd.DataFrame(mean_std).T.to_csv("results/summary_mean_std.csv", index_label="metric")
    
    with open("results/summary_mean_std.json", "w") as f:
        json.dump(mean_std, f, indent=4)
    
    # Create final results dictionary with all details
    results_summary = {}
    for metric_name, values in all_metrics.items():
        if values:
            results_summary[f'{metric_name}_mean'] = np.mean(values)
            results_summary[f'{metric_name}_std'] = np.std(values)
            results_summary[f'{metric_name}_min'] = np.min(values)
            results_summary[f'{metric_name}_max'] = np.max(values)
    
    final_results = {
        'experiment_info': {
            'n_folds': n_folds,
            'min_epochs': min_epochs,
            'max_epochs': max_epochs,
            'patience': patience,
            'batch_size': batch_size,
            'train_split': 0.72,  # ~72% (80% * 90%)
            'val_split': 0.08,    # ~8% (80% * 10%)  
            'test_split': 0.2,
            'input_shape': list(X.shape),
            'num_classes': num_classes
        },
        'summary_statistics': results_summary,
        'mean_std': mean_std,
        'individual_runs': all_runs_results,
        'raw_metrics': all_metrics
    }
    
    # Save comprehensive results to JSON
    with open('results/pathcnn_all_results.json', 'w') as f:
        json.dump(final_results, f, indent=4)
    
    print(f"\nAll results saved in 'results' directory:")
    print(f"  - results/summary_mean_std.csv")
    print(f"  - results/summary_mean_std.json")
    print(f"  - results/pathcnn_all_results.json (comprehensive)")
    print(f"  - results/run_[1-{n_folds}]_results.json (individual folds)")
    
    return final_results

def main():
    """
    Main function to run PathCNN pipeline
    """
    print("Starting PathCNN pipeline...")
    
    # Load and preprocess data (now ensures matching genes)
    mutation_data, cnv_data, labels, gene_names, pathway_genes = load_and_preprocess_data()
    
    # Create pathway-gene mapping
    pathway_mapping, pathway_names = create_pathway_gene_mapping(pathway_genes, gene_names)
    
    # Apply PCA to pathways for each omics type (now uses same gene_names for both)
    mutation_pca, pathway_list = apply_pca_to_pathways(mutation_data, pathway_mapping, gene_names)
    cnv_pca, _ = apply_pca_to_pathways(cnv_data, pathway_mapping, gene_names)
    
    # Create pathway images
    pathway_images = create_pathway_images(mutation_pca, cnv_pca, n_pc=2)
    
    # Order pathways by correlation
    ordered_pathway_images, pathway_order = order_pathways_by_correlation(pathway_images)
    
    print(f"Final pathway images shape: {ordered_pathway_images.shape}")
    
    # Train PathCNN with multiple runs
    final_results = train_pathcnn_multiple_runs(ordered_pathway_images, labels, n_folds=5)
    
    print(f"\nPathCNN pipeline completed!")
    
    return final_results

if __name__ == "__main__":
    main()