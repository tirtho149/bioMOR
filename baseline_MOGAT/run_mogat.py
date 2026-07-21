#!/usr/bin/env python3
"""
Complete pipeline to run MOGAT on CNV, Mutation, and Label data
This script will:
1. Preprocess your data
2. Run the adapted MOGAT model
3. Generate results
"""

import os
import sys
import subprocess
import argparse

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✓ SUCCESS")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("✗ FAILED")
        print("Error:", e.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description='Complete MOGAT pipeline for CNV/Mutation data')
    parser.add_argument('--cnv_file', type=str, required=True, help='Path to CNV CSV file')
    parser.add_argument('--mutation_file', type=str, required=True, help='Path to mutation CSV file') 
    parser.add_argument('--label_file', type=str, required=True, help='Path to label CSV file')
    parser.add_argument('--dataset_name', type=str, default='your_dataset', help='Dataset name for organization')
    parser.add_argument('--skip_preprocessing', action='store_true', help='Skip preprocessing if data already processed')
    
    args = parser.parse_args()
    
    print("MOGAT Pipeline for CNV/Mutation Binary Classification")
    print("="*60)
    
    # Check if input files exist
    for file_path in [args.cnv_file, args.mutation_file, args.label_file]:
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            sys.exit(1)
    
    # Step 1: Data Preprocessing
    if not args.skip_preprocessing:
        preprocess_cmd = f"python data_preprocessor.py --cnv_file {args.cnv_file} --mutation_file {args.mutation_file} --label_file {args.label_file} --output_dir data/{args.dataset_name}"
        
        if not run_command(preprocess_cmd, "Data Preprocessing"):
            print("Preprocessing failed. Exiting.")
            sys.exit(1)
    else:
        print("Skipping preprocessing (--skip_preprocessing flag used)")
    
    # Step 2: Run MOGAT
    mogat_cmd = f"python mogat_adapted.py -data {args.dataset_name}"
    
    if not run_command(mogat_cmd, "Running MOGAT Model"):
        print("MOGAT model failed. Exiting.")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Results available in timestamped directory: MOGAT_{args.dataset_name}_5fold_results_YYYYMMDD_HHMMSS/")
    print("Key output files:")
    print(f"  - MOGAT_{args.dataset_name}_5fold_complete_results.json")
    print(f"  - MOGAT_{args.dataset_name}_5fold_summary_only.json")
    print(f"  - MOGAT_{args.dataset_name}_5fold_summary.csv")
    print(f"  - MOGAT_{args.dataset_name}_5fold_summary.txt")
    print(f"  - MOGAT_{args.dataset_name}_5fold_test_predictions.csv (columns: fold, sample, y_label, y_score)")

if __name__ == "__main__":
    main()
