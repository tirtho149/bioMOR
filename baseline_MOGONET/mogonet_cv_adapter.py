"""
MOGONET Data Adapter with 5-Fold Cross-Validation
Implements 5-fold CV: 1 fold (20%) test, 4 folds (80%) train with 10% validation split
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import argparse

class MOGONETDataAdapterCV:
    def __init__(self, cnv_file, mutation_file, label_file, output_dir='stad_adapted', 
                 n_folds=5, random_state=42):
        self.cnv_file = cnv_file
        self.mutation_file = mutation_file
        self.label_file = label_file
        self.output_dir = output_dir
        self.n_folds = n_folds
        self.random_state = random_state
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def load_and_process_data(self):
        """Load and process the data"""
        print("Loading data files...")
        
        # Load CNV data
        cnv_df = pd.read_csv(self.cnv_file)
        print(f"CNV data shape: {cnv_df.shape}")
        
        # Load mutation data
        mut_df = pd.read_csv(self.mutation_file)
        print(f"Mutation data shape: {mut_df.shape}")
        
        # Load labels
        label_df = pd.read_csv(self.label_file)
        print(f"Label data shape: {label_df.shape}")
        
        # Get patient ID columns (assuming first column)
        cnv_patient_col = cnv_df.columns[0]
        mut_patient_col = mut_df.columns[0]
        label_patient_col = label_df.columns[0]
        
        # Find common patients
        cnv_patients = set(cnv_df[cnv_patient_col].astype(str))
        mut_patients = set(mut_df[mut_patient_col].astype(str))
        label_patients = set(label_df[label_patient_col].astype(str))
        
        common_patients = list(cnv_patients & mut_patients & label_patients)
        print(f"Common patients: {len(common_patients)}")
        
        # Filter to common patients
        cnv_filtered = cnv_df[cnv_df[cnv_patient_col].astype(str).isin(common_patients)].copy()
        mut_filtered = mut_df[mut_df[mut_patient_col].astype(str).isin(common_patients)].copy()
        label_filtered = label_df[label_df[label_patient_col].astype(str).isin(common_patients)].copy()
        
        # Sort by patient ID for consistent ordering
        cnv_filtered = cnv_filtered.sort_values(cnv_patient_col).reset_index(drop=True)
        mut_filtered = mut_filtered.sort_values(mut_patient_col).reset_index(drop=True)
        label_filtered = label_filtered.sort_values(label_patient_col).reset_index(drop=True)
        
        # Extract features (exclude patient ID)
        self.cnv_features = cnv_filtered.drop(columns=[cnv_patient_col]).values.astype(float)
        self.mut_features = mut_filtered.drop(columns=[mut_patient_col]).values.astype(float)
        
        # Get labels (assuming second column contains labels)
        self.labels = label_filtered.iloc[:, 1].values.astype(int)
        self.sample_ids = label_filtered[label_patient_col].astype(str).values
        
        # Get feature names
        self.cnv_feature_names = list(cnv_filtered.drop(columns=[cnv_patient_col]).columns)
        self.mut_feature_names = list(mut_filtered.drop(columns=[mut_patient_col]).columns)
        
        print(f"CNV features shape: {self.cnv_features.shape}")
        print(f"Mutation features shape: {self.mut_features.shape}")
        print(f"Labels shape: {self.labels.shape}")
        print(f"Label distribution: {np.bincount(self.labels)}")
        
    def binarize_mutation_data(self, threshold=0):
        """Binarize mutation data to {0, 1}"""
        print(f"Binarizing mutation data with threshold {threshold}")
        
        # Check current values
        unique_values = np.unique(self.mut_features)
        print(f"Unique mutation values before binarization: {unique_values}")
        
        # Binarize: values > threshold become 1, others become 0
        self.mut_features = (self.mut_features > threshold).astype(float)
        
        # Verify binarization
        unique_after = np.unique(self.mut_features)
        print(f"Unique mutation values after binarization: {unique_after}")
        
    def normalize_features(self):
        """Normalize CNV features, keep mutations binary"""
        print("Normalizing features...")
        
        # Normalize CNV features to [0, 1]
        scaler = StandardScaler()
        cnv_normalized = scaler.fit_transform(self.cnv_features)
        
        # Scale to [0, 1] range
        cnv_min = cnv_normalized.min(axis=0)
        cnv_max = cnv_normalized.max(axis=0)
        cnv_range = cnv_max - cnv_min
        cnv_range[cnv_range == 0] = 1  # Avoid division by zero
        self.cnv_features = (cnv_normalized - cnv_min) / cnv_range
        
        # Mutations are already binary, no normalization needed
        print("Feature normalization completed")
        
    def create_cv_splits(self):
        """Create 5-fold cross-validation splits"""
        print(f"\nCreating {self.n_folds}-fold cross-validation splits...")
        print(f"Random state: {self.random_state}")
        
        # Create stratified k-fold
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        self.fold_splits = []
        for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(self.cnv_features, self.labels)):
            # Further split train_val into train and validation (90%-10% of the 80%)
            # This means validation is 10% of 80% = 8% of total data
            train_labels = self.labels[train_val_idx]
            
            # Calculate validation size: 10% of training set
            val_size = int(len(train_val_idx) * 0.1)
            
            # Use stratified sampling for validation split
            np.random.seed(self.random_state + fold_idx)  # Different seed per fold
            
            # Stratified split
            class_0_idx = train_val_idx[train_labels == 0]
            class_1_idx = train_val_idx[train_labels == 1]
            
            # Sample proportionally from each class
            val_size_c0 = int(len(class_0_idx) * 0.1)
            val_size_c1 = int(len(class_1_idx) * 0.1)
            
            np.random.shuffle(class_0_idx)
            np.random.shuffle(class_1_idx)
            
            val_idx_c0 = class_0_idx[:val_size_c0]
            val_idx_c1 = class_1_idx[:val_size_c1]
            val_idx = np.concatenate([val_idx_c0, val_idx_c1])
            
            train_idx_c0 = class_0_idx[val_size_c0:]
            train_idx_c1 = class_1_idx[val_size_c1:]
            train_idx = np.concatenate([train_idx_c0, train_idx_c1])
            
            self.fold_splits.append({
                'fold': fold_idx + 1,
                'train_idx': train_idx,
                'val_idx': val_idx,
                'test_idx': test_idx
            })
            
            print(f"\nFold {fold_idx + 1}:")
            print(f"  Train samples: {len(train_idx)} ({len(train_idx)/len(self.labels)*100:.1f}%)")
            print(f"  Val samples:   {len(val_idx)} ({len(val_idx)/len(self.labels)*100:.1f}%)")
            print(f"  Test samples:  {len(test_idx)} ({len(test_idx)/len(self.labels)*100:.1f}%)")
            print(f"  Train labels:  {np.bincount(self.labels[train_idx])}")
            print(f"  Val labels:    {np.bincount(self.labels[val_idx])}")
            print(f"  Test labels:   {np.bincount(self.labels[test_idx])}")
        
    def save_fold_data(self, fold_info):
        """Save data for a specific fold"""
        fold_num = fold_info['fold']
        train_idx = fold_info['train_idx']
        val_idx = fold_info['val_idx']
        test_idx = fold_info['test_idx']
        
        fold_dir = os.path.join(self.output_dir, f'fold_{fold_num}')
        os.makedirs(fold_dir, exist_ok=True)
        
        # Save training data
        np.savetxt(os.path.join(fold_dir, "1_tr.csv"), 
                   self.cnv_features[train_idx], delimiter=',')
        np.savetxt(os.path.join(fold_dir, "2_tr.csv"), 
                   self.mut_features[train_idx], delimiter=',')
        np.savetxt(os.path.join(fold_dir, "labels_tr.csv"), 
                   self.labels[train_idx], delimiter=',')
        np.savetxt(os.path.join(fold_dir, "samples_tr.csv"),
                   self.sample_ids[train_idx], fmt='%s', delimiter=',')
        
        # Save validation data
        np.savetxt(os.path.join(fold_dir, "1_val.csv"), 
                   self.cnv_features[val_idx], delimiter=',')
        np.savetxt(os.path.join(fold_dir, "2_val.csv"), 
                   self.mut_features[val_idx], delimiter=',')
        np.savetxt(os.path.join(fold_dir, "labels_val.csv"), 
                   self.labels[val_idx], delimiter=',')
        np.savetxt(os.path.join(fold_dir, "samples_val.csv"),
                   self.sample_ids[val_idx], fmt='%s', delimiter=',')
        
        # Save test data
        np.savetxt(os.path.join(fold_dir, "1_te.csv"), 
                   self.cnv_features[test_idx], delimiter=',')
        np.savetxt(os.path.join(fold_dir, "2_te.csv"), 
                   self.mut_features[test_idx], delimiter=',')
        np.savetxt(os.path.join(fold_dir, "labels_te.csv"), 
                   self.labels[test_idx], delimiter=',')
        np.savetxt(os.path.join(fold_dir, "samples_te.csv"),
                   self.sample_ids[test_idx], fmt='%s', delimiter=',')
        
        # Save feature names (same for all folds)
        with open(os.path.join(fold_dir, "1_featname.csv"), 'w') as f:
            for name in self.cnv_feature_names:
                f.write(f"{name}\n")
                
        with open(os.path.join(fold_dir, "2_featname.csv"), 'w') as f:
            for name in self.mut_feature_names:
                f.write(f"{name}\n")
        
        return fold_dir
        
    def save_all_folds(self):
        """Save data for all folds"""
        print("\nSaving data for all folds...")
        
        for fold_info in self.fold_splits:
            fold_dir = self.save_fold_data(fold_info)
            print(f"  Fold {fold_info['fold']} saved to {fold_dir}")
        
        # Save fold configuration for reference
        config_path = os.path.join(self.output_dir, 'cv_config.txt')
        with open(config_path, 'w') as f:
            f.write(f"Cross-Validation Configuration\n")
            f.write(f"{'='*60}\n")
            f.write(f"Number of folds: {self.n_folds}\n")
            f.write(f"Random state: {self.random_state}\n")
            f.write(f"Total samples: {len(self.labels)}\n")
            f.write(f"CNV features: {self.cnv_features.shape[1]}\n")
            f.write(f"Mutation features: {self.mut_features.shape[1]}\n")
            f.write(f"Classes: {len(np.unique(self.labels))}\n")
            f.write(f"\nFold Details:\n")
            for fold_info in self.fold_splits:
                f.write(f"\nFold {fold_info['fold']}:\n")
                f.write(f"  Train: {len(fold_info['train_idx'])} samples\n")
                f.write(f"  Val:   {len(fold_info['val_idx'])} samples\n")
                f.write(f"  Test:  {len(fold_info['test_idx'])} samples\n")
        
        print(f"\nConfiguration saved to {config_path}")
        
    def process_all(self, mutation_threshold=0):
        """Run the complete processing pipeline"""
        self.load_and_process_data()
        self.binarize_mutation_data(threshold=mutation_threshold)
        self.normalize_features()
        self.create_cv_splits()
        self.save_all_folds()
        
        print("\n" + "="*60)
        print("MOGONET 5-FOLD CV DATA PREPARATION COMPLETED!")
        print("="*60)
        print(f"Dataset: {self.output_dir}")
        print(f"Total patients: {len(self.labels)}")
        print(f"CNV features: {self.cnv_features.shape[1]}")
        print(f"Mutation features: {self.mut_features.shape[1]}")
        print(f"Classes: {len(np.unique(self.labels))} (binary classification)")
        print(f"Number of folds: {self.n_folds}")
        print(f"Train/Val/Test split per fold: ~72%/8%/20%")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Adapt CNV/Mutation data for MOGONET with 5-fold CV')
    parser.add_argument('--cnv_file', type=str, default='cnv_data.csv', 
                       help='Path to CNV CSV file')
    parser.add_argument('--mutation_file', type=str, default='mutation_data.csv',
                       help='Path to mutation CSV file')
    parser.add_argument('--label_file', type=str, default='labels.csv',
                       help='Path to label CSV file')
    parser.add_argument('--output_dir', type=str, default='stad_adapted',
                       help='Output directory')
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of CV folds (default: 5)')
    parser.add_argument('--mutation_threshold', type=float, default=0,
                       help='Threshold for binarizing mutations (default: 0)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Create adapter and process data
    adapter = MOGONETDataAdapterCV(
        cnv_file=args.cnv_file,
        mutation_file=args.mutation_file,
        label_file=args.label_file,
        output_dir=args.output_dir,
        n_folds=args.n_folds,
        random_state=args.random_state
    )
    
    adapter.process_all(mutation_threshold=args.mutation_threshold)

if __name__ == "__main__":
    main()
