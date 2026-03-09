#!/usr/bin/env python3
# dedup_nested_dirs.py
"""
Deduplicate ALL FASTA files from your nested directory structure:
train_folder/case1/*.fasta, train_folder/case2/*.fasta, test_folder/*.fasta
Global clustering across ALL files while preserving origin tracking.
"""

import os
import subprocess
from pathlib import Path
from Bio import SeqIO
from collections import defaultdict

def find_all_fasta_files(base_dirs):
    """Recursively find all .fasta/.fa files, track their origin."""
    all_seqs = []
    
    for base_dir in base_dirs:
        base_dir = Path(base_dir)
        print(f"Scanning {base_dir}...")
        
        for fasta_file in base_dir.rglob("*.fasta"):
            case_folder = fasta_file.parent.name  # case1, case2, etc.
            dataset = base_dir.parent.name        # train_folder, test_folder
            
            print(f"Found {fasta_file} ({dataset}/{case_folder})")
            
            # Read sequences with origin prefix
            for record in SeqIO.parse(fasta_file, "fasta"):
                if len(record.seq) < 10000000:
                    record.id = f"{dataset}_{case_folder}_{fasta_file.stem}|{record.id}"
                    record.description += f" [from:{fasta_file}]"
                    all_seqs.append(record)
    
    print(f"\n Found {len(all_seqs)} total sequences")
    return all_seqs

def run_global_dedup(combined_fasta, output_dir, threshold=0.95):
    """Run CD-HIT on combined FASTA."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    nr_fasta = output_dir / "non_redundant.fa"
    clstr_file = output_dir / "non_redundant.clstr"
    
    print(f"Global deduplication at {threshold*100}% identity...")
    
    cmd = [
        "cd-hit-est",           # DNA sequences
        "-i", str(combined_fasta),
        "-o", str(nr_fasta),
        "-c", str(threshold),
        "-n", "8",
        "-M", "0",              # unlimited memory
        "-T", "0"               # all CPU cores
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"CD-HIT failed: {result.stderr}")
        return None
    
    print(f"Non-redundant: {nr_fasta}")
    print(f"Clusters:      {clstr_file}")
    return nr_fasta, clstr_file

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Your directory structure
    BASE_DIRS = [
        #"dataset/train_folder/case1",    # case1/, case2/ subdirs
        "dataset/train_folder/training_pool",      # test.fasta
        "dataset/test_folder"
    ]
    
    OUTPUT_DIR = "dataset/deduplicated_global"
    THRESHOLD = 0.95  # 95% identity clustering
    
    print("=== GLOBAL DEDUPLICATION ===")
    
    # Step 1: Find ALL sequences
    all_seqs = find_all_fasta_files(BASE_DIRS)
    
    # Step 2: Write combined FASTA with origin tracking
    combined_fa = Path(OUTPUT_DIR) / "all_sequences_with_origin.fa"
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    print(f"\n Writing combined FASTA: {combined_fa}")
    SeqIO.write(all_seqs, combined_fa, "fasta")
    
    # Step 3: Run CD-HIT global clustering
    nr_result = run_global_dedup(combined_fa, OUTPUT_DIR, THRESHOLD)
    
    if nr_result:
        print(f"\n SUCCESS!")
        print(f" Original sequences: {len(all_seqs)}")
        print(f" Non-redundant reps: {len(list(SeqIO.parse(nr_result[0], 'fasta')))}")
        print(f"\n Cluster file shows EXACTLY which sequences clustered together:")
        print(f"{nr_result[1]}")
        
        # Show cluster summary
        with open(nr_result[1]) as f:
            clusters = len([line for line in f if line.startswith("Cluster")])
        print(f"{clusters} total clusters created")
