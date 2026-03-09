# import libraries
import os
import numpy as np 
import pandas as pd
import random
from scipy.stats import lognorm
from Bio.SeqUtils import gc_fraction
from Bio import SeqIO
from utils.data_processing import fasta_to_df

FOLDER_TRAIN_DATA = "./dataset/train_folder"

# ============================function for sampling============================
def uniform_caps(meta_df, chosen_species, target_total_samples):
    """
    Given a set of species and a target total number of samples,
    compute how many samples to draw per species (uniform cap).
    """
    sub = meta_df[meta_df['species'].isin(chosen_species)].copy()
    n_species = len(sub)

    # ideal equal allocation (may be fractional)
    ideal_per_species = target_total_samples / n_species

    # cap by availability
    sub['cap'] = sub['n_samples'].clip(upper=np.floor(ideal_per_species))

    # if too few samples overall, just use all available
    total_cap = sub['cap'].sum()
    if total_cap == 0:
        # fall back to at least 1 sample if possible
        sub['cap'] = sub['n_samples'].clip(upper=1)
        total_cap = sub['cap'].sum()

    return sub

def proportional_caps(meta_df, chosen_species, target_total_samples):
    """
    Compute how many samples to take per species so that:
      - species keep their relative abundances as much as possible
      - total ≈ target_total_samples
      - never exceed available 'number_of_samples'.
    """
    sub = meta_df[meta_df['species'].isin(chosen_species)].copy()

    # available samples per species
    avail = sub['n_samples'].values.astype(float)
    total_avail = avail.sum()

    if total_avail <= target_total_samples:
        # Not enough data to hit target: just take everything
        sub['cap'] = sub['n_samples']
        return sub

    # initial proportional scaling factor
    scale = target_total_samples / total_avail

    # proposed draws
    proposed = np.floor(avail * scale).astype(int)
    # make sure we take at least 1 from any species that had samples
    proposed = np.maximum(proposed, 1)
    proposed = np.minimum(proposed, avail.astype(int))  # cannot exceed availability

    # adjust to match target_total_samples as closely as possible
    diff = target_total_samples - proposed.sum()

    # if diff > 0, distribute remaining samples to species with leftover capacity
    if diff > 0:
        leftover_capacity = avail.astype(int) - proposed
        # indices of species that can still give more
        candidates = np.where(leftover_capacity > 0)[0]
        # repeatedly add 1 sample to candidates until diff is exhausted or no capacity
        i = 0
        while diff > 0 and len(candidates) > 0:
            idx = candidates[i % len(candidates)]
            if leftover_capacity[idx] > 0:
                proposed[idx] += 1
                leftover_capacity[idx] -= 1
                diff -= 1
            i += 1

    # if diff < 0, remove some samples proportionally from species with >1
    elif diff < 0:
        to_remove = -diff
        candidates = np.where(proposed > 1)[0]
        i = 0
        while to_remove > 0 and len(candidates) > 0:
            idx = candidates[i % len(candidates)]
            if proposed[idx] > 1:
                proposed[idx] -= 1
                to_remove -= 1
            i += 1

    sub['cap'] = proposed
    return sub

# Overlap checker (no intervaltree)
def overlaps(chrom, start, end, regions_dict):
    """
    Check if a given region overlaps with any regions in the provided dictionary.
    Parameters:
    ----------
    chrom (str): Chromosome name.
    start (int): Start position of the region.
    end (int): End position of the region.
    regions_dict (dict): Dictionary containing regions with chromosome names as keys and lists of tuples (start, end) as values.
    Returns:
    -------
    bool: True if there is an overlap, False otherwise.
    """
    # Check if the chromosome exists in the dictionary
    if chrom not in regions_dict:
        return False
    for a, b in regions_dict[chrom]:
        if start < b and end > a:
            return True
    return False

# Negative sampler
def sample_negative(genome, positives, neg_samples=0, neg_reference=f"{FOLDER_TRAIN_DATA}/islandpick_data.fasta"):
    """
    Sample negative regions from the genome, avoiding overlaps with positive regions.
    Parameters:
    ----------
    genome (dict): Dictionary containing genome sequences with chromosome names as keys.
    positives (dict): Dictionary containing positive regions with chromosome names as keys.
    neg_samples (int): Number of negative samples to generate per chromosome. If 0, uses the number of positives.
    neg_reference (str): Path to the reference FASTA file for negative sample statistics.
    Returns:
    -------
    negatives (list): List of tuples containing negative regions (chromosome, start, end).
    """

    # read islandpick data to calculate statistics of the negative samples
    islandpick_data = fasta_to_df(neg_reference)
    # calculate statistics of the negative samples from islandpick data
    islandpick_data['Length'] = islandpick_data.apply(lambda x: x['End']-x['Start'],axis=1)
    islandpick_data['gc'] = islandpick_data.apply(lambda x: gc_fraction(x['Sequence']),axis=1)
    islandpick_data_neg = islandpick_data[islandpick_data['Label']=='0']
    # learn length distribution from curated dataset of islandpick
    lengths = islandpick_data_neg.Length.to_list()
    # Length distribution fitting (e.g. lognormal)
    shape, loc, scale = lognorm.fit(islandpick_data_neg.Length.to_list(), floc=0)
    # Parameters
    min_len = min(lengths)
    max_len = max(lengths)
    
    chromosomes = list(genome.keys())
    all_negatives = []
    for chrom in chromosomes:
        negatives = []
        tries = 0
        #if neg_samples=0, then use the number of positives (e.g. balanced sampling)
        n_samples = neg_samples
        if n_samples == 0:
            n_samples = len(positives[chrom])

        max_tries = n_samples * 50
        while len(negatives) < n_samples and tries < max_tries:
            #chrom = random.choice(chromosomes)
            sequence = genome[chrom]['Sequence']
            chrom_len = len(sequence)

            length = int(np.clip(lognorm.rvs(shape, loc=loc, scale=scale), min_len, max_len))
  
            if chrom_len <= length:
                tries += 1
            else:
                start = random.randint(0, chrom_len - length)
                end = start + length

                # Avoid overlap with positives
                if overlaps(chrom, start, end, positives):
                    tries += 1
                else:
                    negatives.append((chrom, start, end))
    
        all_negatives.extend(negatives)
    return all_negatives

# list samples genomes 
def negatives_sampling(positives, genome_path):
    """
    A function to sample negative regions from genomes in islandviewer4 data.
    Parameters:
    ----------
    positives : pandas.DataFrame, DataFrame containing positive regions from islandviewer4.
    genome_path : str, path to the directory containing genome fasta files.
    Returns:
    -------
    final_df : pandas.DataFrame, DataFrame containing both positive and negative samples.
    """

    def read_file(file_path):
        seq_records = {}
        for seq_record in SeqIO.parse(file_path, "fasta"):
            id = seq_record.id
            desc = seq_record.description
            sequence = str(seq_record.seq)
            seq_records.update({id:{'Sequence':sequence,'Description':desc}})

        return seq_records

    # read samples genomes 
    genomes_dict = {}

    for file in os.listdir(genome_path):
        file_path = os.path.join(genome_path,file)
        if os.path.isfile(file_path) and file_path.endswith((".fasta",".fa",".fna")):
            record = read_file(file_path)
            genomes_dict.update(record)
        elif os.path.isdir(file_path):
            for file in os.listdir(file_path):
                childfile_path = os.path.join(file_path, file) 
                if os.path.isfile(childfile_path) and childfile_path.endswith((".fasta",".fa",".fna")):
                    record = read_file(childfile_path)
                    genomes_dict.update(record)


    # select sample genomes from positives data
    selected_ids = set(genomes_dict.keys()) & set(positives.accession.unique())
    pos_df = positives[positives['accession'].isin(selected_ids)].reset_index(drop=True)
    pos_df['Label'] = '1'

    # collect positive regions per genome
    genome = {k:genomes_dict[k] for k in selected_ids}
    positive_regions = pos_df.groupby('accession').apply(lambda g: list(zip(g['start'], g['end']))).to_dict()

    del genomes_dict

    # run negative sampling
    negatives_samples = sample_negative(genome, positive_regions)
    neg_df = pd.DataFrame(negatives_samples, columns=["accession", "start", "end"]) 
    neg_df['Label'] = '0'
    neg_df['prediction_method'] = 'random_neg_sample'

    #combine pos and neg samples
    final_df = pd.concat([pos_df,neg_df])
    final_df['Sequence'] = final_df.apply(lambda x: genome[x['accession']]['Sequence'][x['start']:x['end']+1],axis=1)
    final_df['Description'] = final_df.apply(lambda x: f"{x['accession']}:{x['start']}-{x['end']} {genome[x['accession']]['Description']}", axis=1)
    final_df['gc'] = final_df.apply(lambda x: gc_fraction(x['Sequence']),axis=1)
    final_df.rename(columns={'accession':'Accession','start':'Start','end':'End'},inplace=True)

    return final_df