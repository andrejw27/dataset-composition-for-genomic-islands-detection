import os
import numpy as np
import pandas as pd
import subprocess
from Bio import SeqIO
from Bio.Seq import Seq
from scipy.spatial.distance import jensenshannon
from sklearn.feature_extraction.text import CountVectorizer

def get_canonical_kmer_probs(fasta_path, k=4):
    """Parses a FASTA file and returns a canonical k-mer probability vector."""
    sequences = [str(rec.seq).upper() for rec in SeqIO.parse(fasta_path, "fasta")]
    
    # 1. Create a generator for canonical k-mers
    def kmer_gen(seqs, k_len):
        for seq in seqs:
            for i in range(len(seq) - k_len + 1):
                kmer = seq[i:i+k_len]
                if 'N' not in kmer:
                    rc = str(Seq(kmer).reverse_complement())
                    yield kmer if kmer < rc else rc

    # 2. Join kmers into a 'sentence' for the vectorizer
    kmers_str = " ".join(kmer_gen(sequences, k))
    return kmers_str

def cross_folder_jsd(folder_a, folder_b, k=4):
    # Get all fasta files
    files_a = [f for f in os.listdir(folder_a) if f.endswith(('.fasta', '.fa'))]
    files_b = [f for f in os.listdir(folder_b) if f.endswith(('.fasta', '.fa'))]
    
    # Combine everything to build a unified vocabulary
    all_names = files_a + files_b
    all_paths = [os.path.join(folder_a, f) for f in files_a] + \
                [os.path.join(folder_b, f) for f in files_b]
    
    print(f"Processing {len(all_paths)} files at k={k}...")
    
    # 3. Vectorize all files at once to ensure aligned indices
    corpus = [get_canonical_kmer_probs(p, k) for p in all_paths]
    vectorizer = CountVectorizer(analyzer='word', token_pattern=r"(?u)\b\w+\b")
    counts = vectorizer.fit_transform(corpus).toarray()
    
    # 4. Convert to probabilities
    probs = counts / (counts.sum(axis=1)[:, None] + 1e-10) # Added epsilon to avoid div by zero
    
    # Split the results back into A and B
    probs_a = probs[:len(files_a)]
    probs_b = probs[len(files_a):]
    
    # 5. Build the comparison matrix
    matrix = np.zeros((len(files_a), len(files_b)))
    for i in range(len(files_a)):
        for j in range(len(files_b)):
            # Squared distance = Divergence
            matrix[i, j] = jensenshannon(probs_a[i], probs_b[j])
            
    return pd.DataFrame(matrix, index=files_a, columns=files_b)

def get_fasta_paths(directory):
    valid_exts = (".fasta", ".fa", ".fna")
    return [
        os.path.abspath(os.path.join(directory, f)) 
        for f in os.listdir(directory) 
        if f.endswith(valid_exts)
    ]

def compute_mash_distance(reference, query, k="21", s="10000", output_file="mash_distances.txt"):

    files_a = get_fasta_paths(query)
    files_b = get_fasta_paths(reference)

    # Create Mash "sketches" for Folder B (Reference)
    print("Sketching genomes in Reference...")
    subprocess.run([
        "mash", "sketch", 
        "-o", "reference_sketch", 
        "-k", k,
        "-s", "10000",
        *files_b
    ], check=True)

    # Calculate distances for each genome in Folder A (Query) against the sketch
    print("Calculating Mash distances...")
    with open(output_file, "w") as out_f:
        for query_file in files_a:
            # Run Mash distance
            result = subprocess.run(
                ["mash", "dist", "reference_sketch.msh", query_file],
                capture_output=True, text=True, check=True
            )
            out_f.write(result.stdout)

    # Clean up the sketch file
    if os.path.exists("reference_sketch.msh"):
        os.remove("reference_sketch.msh")

if __name__ == "__main__":
    # --- EXECUTION ---
    # Add your file paths here
    FOLDER_PATH_TRAIN = "dataset/deduplicated_data/data"
    FOLDER_PATH_TEST = "dataset/test_folder"

    distance_metric = "jsd"
    
    if distance_metric == "jsd":
        result_df = cross_folder_jsd(FOLDER_PATH_TRAIN, FOLDER_PATH_TEST, k=7)
        result_df.to_excel(f"{distance_metric}.xlsx")
    elif distance_metric == "mash":
        compute_mash_distance(FOLDER_PATH_TRAIN, FOLDER_PATH_TEST, output_file="mash_distances.txt")
    else:
        raise ValueError("Unsupported distance metric. Use 'jsd' or 'mash'.")

    