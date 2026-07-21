# Data Preprocessor for CNV, Mutation, and Label datasets
# Adapts your data format to MOGAT-compatible format
# Updated for compatibility with multiple random state runs

import pandas as pd
import numpy as np
import pickle
import os
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse

class DataPreprocessor:
    def __init__(self, cnv_file, mutation_file, label_file, output_dir='data/your_dataset'):
        self.cnv_file = cnv_file
        self.mutation_file = mutation_file
        self.label_file = label_file
        self.output_dir = output_dir
        # Removed fixed random_state since MOGAT will handle splits dynamically
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def load_data(self):
        """Load CNV, mutation, and label data"""
        print("Loading data files...")
        
        # Load CNV data
        self.cnv_df = pd.read_csv(self.cnv_file)
        print(f"CNV data shape: {self.cnv_df.shape}")
        
        # Load mutation data  
        self.mut_df = pd.read_csv(self.mutation_file)
        print(f"Mutation data shape: {self.mut_df.shape}")
        
        # Load labels
        self.label_df = pd.read_csv(self.label_file)
        print(f"Label data shape: {self.label_df.shape}")
        
        # Get patient ID column names (assuming first column)
        self.cnv_patient_col = self.cnv_df.columns[0]
        self.mut_patient_col = self.mut_df.columns[0]  
        self.label_patient_col = self.label_df.columns[0]
        
        print(f"Patient ID columns: CNV={self.cnv_patient_col}, MUT={self.mut_patient_col}, LABEL={self.label_patient_col}")
        
    def find_common_patients(self):
        """Find patients common across all datasets"""
        cnv_patients = set(self.cnv_df[self.cnv_patient_col].astype(str))
        mut_patients = set(self.mut_df[self.mut_patient_col].astype(str))
        label_patients = set(self.label_df[self.label_patient_col].astype(str))
        
        # Find intersection
        self.common_patients = list(cnv_patients & mut_patients & label_patients)
        print(f"Common patients across all datasets: {len(self.common_patients)}")
        
        if len(self.common_patients) < 10:
            raise ValueError("Too few common patients found. Check patient ID matching.")
            
    def filter_common_patients(self):
        """Filter datasets to keep only common patients"""
        # Convert patient IDs to string for consistent matching
        self.cnv_df[self.cnv_patient_col] = self.cnv_df[self.cnv_patient_col].astype(str)
        self.mut_df[self.mut_patient_col] = self.mut_df[self.mut_patient_col].astype(str)
        self.label_df[self.label_patient_col] = self.label_df[self.label_patient_col].astype(str)
        
        # Filter datasets
        self.cnv_filtered = self.cnv_df[self.cnv_df[self.cnv_patient_col].isin(self.common_patients)].copy()
        self.mut_filtered = self.mut_df[self.mut_df[self.mut_patient_col].isin(self.common_patients)].copy()
        self.label_filtered = self.label_df[self.label_df[self.label_patient_col].isin(self.common_patients)].copy()
        
        # Sort by patient ID to ensure consistent ordering
        self.cnv_filtered = self.cnv_filtered.sort_values(self.cnv_patient_col).reset_index(drop=True)
        self.mut_filtered = self.mut_filtered.sort_values(self.mut_patient_col).reset_index(drop=True)
        self.label_filtered = self.label_filtered.sort_values(self.label_patient_col).reset_index(drop=True)
        
        print(f"Filtered CNV shape: {self.cnv_filtered.shape}")
        print(f"Filtered mutation shape: {self.mut_filtered.shape}")
        print(f"Filtered label shape: {self.label_filtered.shape}")
        
    def prepare_features(self):
        """Prepare feature matrices and handle missing values"""
        # CNV features (exclude patient ID column)
        cnv_features = self.cnv_filtered.drop(columns=[self.cnv_patient_col])
        
        # Handle missing values in CNV (fill with 0 - no change)
        cnv_features = cnv_features.fillna(0)
        
        # Ensure CNV values are in expected range [-2, -1, 0, 1, 2]
        cnv_values = cnv_features.values.flatten()
        unique_cnv = np.unique(cnv_values[~np.isnan(cnv_values)])
        print(f"Unique CNV values: {unique_cnv}")
        
        # Mutation features (exclude patient ID column)
        mut_features = self.mut_filtered.drop(columns=[self.mut_patient_col])
        
        # Handle missing values in mutations (fill with 0 - no mutation)
        mut_features = mut_features.fillna(0)
        
        # Ensure mutation values are binary
        mut_values = mut_features.values.flatten()
        unique_mut = np.unique(mut_values[~np.isnan(mut_values)])
        print(f"Unique mutation values: {unique_mut}")
        
        # Convert to numpy arrays
        self.cnv_matrix = cnv_features.values.astype(float)
        self.mut_matrix = mut_features.values.astype(float)
        
        # Get gene names
        self.cnv_genes = list(cnv_features.columns)
        self.mut_genes = list(mut_features.columns)
        
        print(f"CNV matrix shape: {self.cnv_matrix.shape}")
        print(f"Mutation matrix shape: {self.mut_matrix.shape}")
        print(f"Number of CNV genes: {len(self.cnv_genes)}")
        print(f"Number of mutation genes: {len(self.mut_genes)}")
        
    def prepare_labels(self):
        """Prepare labels"""
        # Get labels (assuming second column contains labels)
        label_col = self.label_filtered.columns[1]
        self.labels = self.label_filtered[label_col].values.astype(int)
        
        print(f"Labels shape: {self.labels.shape}")
        print(f"Label distribution: {np.bincount(self.labels)}")
        
        # Verify labels are binary (0, 1)
        unique_labels = np.unique(self.labels)
        if not np.array_equal(unique_labels, [0, 1]):
            print(f"Warning: Labels are not binary. Found: {unique_labels}")
            if len(unique_labels) == 2:
                print("Converting to binary (0, 1)")
                # Map to 0, 1
                label_mapping = {unique_labels[0]: 0, unique_labels[1]: 1}
                self.labels = np.array([label_mapping[label] for label in self.labels])
                print(f"Converted label distribution: {np.bincount(self.labels)}")
        
    def normalize_features(self):
        """Normalize CNV features (mutations are already binary)"""
        # Normalize CNV features
        scaler = StandardScaler()
        self.cnv_matrix_norm = scaler.fit_transform(self.cnv_matrix)
        
        # Mutations don't need normalization (binary)
        self.mut_matrix_norm = self.mut_matrix.copy()
        
        print("Feature normalization completed")
        print(f"CNV features - Mean: {self.cnv_matrix_norm.mean():.4f}, Std: {self.cnv_matrix_norm.std():.4f}")
        
    def save_mogat_format(self):
        """Save data in MOGAT-compatible format"""
        print("Saving data in MOGAT format...")
        
        # Save CNV data as pandas DataFrame (MOGAT expects this format)
        cnv_df_for_mogat = pd.DataFrame(self.cnv_matrix_norm, columns=self.cnv_genes)
        with open(os.path.join(self.output_dir, 'cna.pkl'), 'wb') as f:
            pickle.dump(cnv_df_for_mogat, f)
            
        # Save mutation data as pandas DataFrame  
        mut_df_for_mogat = pd.DataFrame(self.mut_matrix_norm, columns=self.mut_genes)
        with open(os.path.join(self.output_dir, 'mut.pkl'), 'wb') as f:
            pickle.dump(mut_df_for_mogat, f)
            
        # Save labels
        with open(os.path.join(self.output_dir, 'labels.pkl'), 'wb') as f:
            pickle.dump(self.labels, f)

        # Save sample IDs in the same order as labels for downstream fold-level reporting
        sample_ids = self.label_filtered[self.label_patient_col].astype(str).tolist()
        with open(os.path.join(self.output_dir, 'sample_ids.pkl'), 'wb') as f:
            pickle.dump(sample_ids, f)
            
        # NOTE: No longer saving mask_values.pkl since MOGAT will create splits dynamically
        print("Note: Train/test splits will be created dynamically by MOGAT with different random states")
            
        # Create edges (patient similarity networks) for each data type
        self.create_patient_similarity_networks()
        
        print(f"Data saved to {self.output_dir}")
        
    def create_patient_similarity_networks(self):
        """Create patient similarity networks for CNV and mutation data"""
        from scipy.spatial.distance import pdist, squareform
        from scipy.stats import pearsonr
        import networkx as nx
        
        # For CNV data - use Pearson correlation
        print("Creating CNV patient similarity network...")
        cnv_corr_matrix = np.corrcoef(self.cnv_matrix_norm)
        
        # Handle NaN values in correlation matrix
        cnv_corr_matrix = np.nan_to_num(cnv_corr_matrix, nan=0.0)
        
        cnv_edges = self.create_edges_from_similarity(cnv_corr_matrix, top_k=5)  # Increased from 3 to 5
        
        cnv_edges_df = pd.DataFrame(cnv_edges, columns=['source', 'target', 'weight'])
        with open(os.path.join(self.output_dir, 'edges_cna.pkl'), 'wb') as f:
            pickle.dump(cnv_edges_df, f)
            
        print(f"CNV network: {len(cnv_edges)} edges created")
            
        # For mutation data - use Jaccard similarity
        print("Creating mutation patient similarity network...")
        mut_jaccard_matrix = self.jaccard_similarity_matrix(self.mut_matrix_norm)
        mut_edges = self.create_edges_from_similarity(mut_jaccard_matrix, top_k=5)  # Increased from 3 to 5
        
        mut_edges_df = pd.DataFrame(mut_edges, columns=['source', 'target', 'weight'])
        with open(os.path.join(self.output_dir, 'edges_mut.pkl'), 'wb') as f:
            pickle.dump(mut_edges_df, f)
            
        print(f"Mutation network: {len(mut_edges)} edges created")
        print("Patient similarity networks created")
        
    def jaccard_similarity_matrix(self, binary_matrix):
        """Calculate Jaccard similarity matrix for binary data"""
        n_samples = binary_matrix.shape[0]
        jaccard_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i, n_samples):
                if i == j:
                    jaccard_matrix[i, j] = 1.0
                else:
                    # Calculate Jaccard similarity
                    intersection = np.sum((binary_matrix[i] == 1) & (binary_matrix[j] == 1))
                    union = np.sum((binary_matrix[i] == 1) | (binary_matrix[j] == 1))
                    
                    if union == 0:
                        jaccard_sim = 0.0
                    else:
                        jaccard_sim = intersection / union
                        
                    jaccard_matrix[i, j] = jaccard_sim
                    jaccard_matrix[j, i] = jaccard_sim
                    
        return jaccard_matrix
        
    def create_edges_from_similarity(self, similarity_matrix, top_k=5):
        """Create edges by selecting top-k similar patients for each patient"""
        n_patients = similarity_matrix.shape[0]
        edges = []
        
        for i in range(n_patients):
            # Get similarities for patient i (exclude self)
            similarities = similarity_matrix[i].copy()
            similarities[i] = -np.inf  # Exclude self-connection
            
            # Get top-k most similar patients
            top_indices = np.argsort(similarities)[-top_k:]
            
            for j in top_indices:
                if similarities[j] > 0:  # Only add positive similarities
                    edges.append([i, j, similarities[j]])
                    
        return edges
        
    def validate_data_quality(self):
        """Validate the quality of processed data"""
        print("\n" + "="*50)
        print("DATA QUALITY VALIDATION")
        print("="*50)
        
        # Check for class balance
        class_counts = np.bincount(self.labels)
        class_ratio = min(class_counts) / max(class_counts)
        print(f"Class balance ratio: {class_ratio:.3f}")
        if class_ratio < 0.3:
            print("Warning: Severe class imbalance detected")
        
        # Check feature statistics
        print(f"\nCNV features:")
        print(f"  - Range: [{self.cnv_matrix_norm.min():.3f}, {self.cnv_matrix_norm.max():.3f}]")
        print(f"  - Missing values: {np.isnan(self.cnv_matrix_norm).sum()}")
        
        print(f"\nMutation features:")
        print(f"  - Range: [{self.mut_matrix_norm.min():.3f}, {self.mut_matrix_norm.max():.3f}]")
        print(f"  - Missing values: {np.isnan(self.mut_matrix_norm).sum()}")
        print(f"  - Sparsity: {(self.mut_matrix_norm == 0).mean():.3f}")
        
        # Check for constant features
        cnv_constant = np.sum(np.std(self.cnv_matrix_norm, axis=0) == 0)
        mut_constant = np.sum(np.std(self.mut_matrix_norm, axis=0) == 0)
        
        print(f"\nConstant features:")
        print(f"  - CNV: {cnv_constant} ({cnv_constant/self.cnv_matrix_norm.shape[1]*100:.1f}%)")
        print(f"  - Mutation: {mut_constant} ({mut_constant/self.mut_matrix_norm.shape[1]*100:.1f}%)")
        
        print("="*50)
        
    def process_all(self):
        """Run the complete preprocessing pipeline"""
        self.load_data()
        self.find_common_patients()
        self.filter_common_patients()
        self.prepare_features()
        self.prepare_labels()
        self.normalize_features()
        self.validate_data_quality()
        self.save_mogat_format()
        
        print("\nData preprocessing completed successfully!")
        print(f"Final dataset info:")
        print(f"- Number of patients: {len(self.common_patients)}")
        print(f"- CNV features: {self.cnv_matrix.shape[1]}")
        print(f"- Mutation features: {self.mut_matrix.shape[1]}")
        print(f"- Labels: {len(np.unique(self.labels))} classes")
        print(f"- Output directory: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess CNV, Mutation, and Label data for MOGAT')
    parser.add_argument('--cnv_file', type=str, required=True, help='Path to CNV CSV file')
    parser.add_argument('--mutation_file', type=str, required=True, help='Path to mutation CSV file')
    parser.add_argument('--label_file', type=str, required=True, help='Path to label CSV file')
    parser.add_argument('--output_dir', type=str, default='data/your_dataset', help='Output directory')
    
    args = parser.parse_args()
    
    preprocessor = DataPreprocessor(
        cnv_file=args.cnv_file,
        mutation_file=args.mutation_file,
        label_file=args.label_file,
        output_dir=args.output_dir
    )
    
    preprocessor.process_all()

if __name__ == "__main__":
    main()
