import os
import numpy as np 
import pandas as pd
import random
from tqdm import tqdm
import json
import pickle
from scipy.stats import lognorm
from Bio import SeqIO, Entrez
from Bio.SeqUtils import gc_fraction
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, matthews_corrcoef,confusion_matrix, accuracy_score, precision_score, f1_score, fbeta_score, recall_score
from . import FileProcessing, CheckAccPseParameter
from sklearn.preprocessing import FunctionTransformer
from .Parameters import Parameters

# Create new `pandas` methods which use `tqdm` progress
# (can use tqdm_gui, optional kwargs, etc.)
tqdm.pandas()

Entrez.email = "A.N.Other@example.com"  # Always tell NCBI who you are

import os, sys, re
from pathlib import Path
file_path = os.path.split(os.path.realpath(__file__))[0]
pPath = Path(file_path).parent
sys.path.append(pPath)

import logging 
logger = logging.getLogger('util')

FOLDER_TRAIN = os.path.join(file_path,'..','dataset','train_folder')

#reference: https://stackoverflow.com/questions/13575090/construct-pandas-dataframe-from-items-in-nested-dictionary
def flatten_dict(nested_dict):
    """
    transform a nested dictionary into a normal dictionary with {key:value} format

    Parameters: 
    ----------
    nested_dict : dict, a nested dictionary

    Returns: 
    -------
    output_dict : dict, flattened dictionary
    """

    output_dict = {}
    if isinstance(nested_dict, dict):
        for k in nested_dict:
            flattened_dict = flatten_dict(nested_dict[k])
            for key, val in flattened_dict.items():
                key = list(key)
                key.insert(0, k)
                output_dict[tuple(key)] = val
    else:
        output_dict[()] = nested_dict
    return output_dict
    
def nested_dict_to_df(input_dict):
    """
    transform a nested dictionary into a dataframe

    Parameters: 
    ----------
    input_dict : dict, a nested dictionary

    Returns: 
    -------
    output_df : pandas.DataFrame
    """
    flat_dict = flatten_dict(input_dict)
    output_df = pd.DataFrame.from_dict(flat_dict, orient="index")
    output_df.index = pd.MultiIndex.from_tuples(output_df.index)
    output_df = output_df.unstack(level=-1)
    output_df.columns = output_df.columns.map("{0[1]}".format)
    return output_df

########################## function to turn multiindex dictionary to dataframe ##########################
def multiindex_dict_to_df(input_dict):
    """
    transform a dictionary with tuples as keys into a dataframe

    Parameters: 
    ----------
    input_dict : dict, a multiindex dictionary, example: {(tuple):value}
    
    Returns: 
    -------
    output_df : pandas.DataFrame
    """
    output_df = pd.DataFrame.from_dict(input_dict, orient="index")
    output_df.index = pd.MultiIndex.from_tuples(output_df.index)
    output_df = output_df.unstack(level=-1)
    output_df.columns = output_df.columns.map("{0[1]}".format)
    return output_df
    
########################## function to encode data set ##########################
def transform_sequences(X, representation, desc_default_para):
    try:
        X,_,_ = get_representations(X, representation, desc_default_para)

        return X
    except Exception as e:
        raise ValueError("Error in get_representations: {}".format(e))

def get_customer_transformer(representation):
    """
    Get the parameters for the specified representation.
    :param representation: The representation type (e.g., 'Z_curve', 'Kmer-6').
    :return: A FunctionTransformer that applies the transformation.
    """
    # parameters for features encoding
    parameters = Parameters()
    desc_default_para = parameters.DESC_DEFAULT_PARA
    para_dict = parameters.PARA_DICT

    if 'Kmer' in representation:
        k = int(representation.split('-')[-1])
        desc_default_para.update({'kmer': k})
        representation = representation.split('-')[0]
    elif representation in para_dict:
        for key in para_dict[representation]:
            desc_default_para.update({key: para_dict[representation][key]}) 

    # Wrap it in a transformer that binds the extra arguments
    custom_transformer = FunctionTransformer(
        func=lambda X: transform_sequences(X, representation=representation, desc_default_para=desc_default_para),
        validate=False
    )

    return custom_transformer

def get_representations(filename, desc, desc_default_para):
    """
    transform a dictionary with tuples as keys into a dataframe

    Parameters: 
    ----------
    desc : str, type of data representation
    filename : str, path to the fasta file
    desc_default_para : dict, parameters for the corresponding data representation, example: k values for kmer

    Returns: 
    -------
    X : array-like, feature matrix used for training. Each row represents a sample and each column a feature.
    y : array-like, target values (class labels) corresponding to the input samples.
    groups : array-like, groups (species) corresponding to the input samples
    """
    descriptor = FileProcessing.Descriptor(filename, desc_default_para)
    
    if desc in ['DAC', 'TAC']:
        my_property_name, my_property_value, my_kmer, ok = CheckAccPseParameter.check_acc_arguments(desc, descriptor.check_sequence_type(), desc_default_para)
        status = descriptor.make_ac_vector(my_property_name, my_property_value, my_kmer)
    elif desc in ['DCC', 'TCC']:
        my_property_name, my_property_value, my_kmer, ok = CheckAccPseParameter.check_acc_arguments(desc, descriptor.check_sequence_type(), desc_default_para)
        status = descriptor.make_cc_vector(my_property_name, my_property_value, my_kmer)
    elif desc in ['DACC', 'TACC']:
        my_property_name, my_property_value, my_kmer, ok = CheckAccPseParameter.check_acc_arguments(desc, descriptor.check_sequence_type(), desc_default_para)
        status = descriptor.make_acc_vector(my_property_name, my_property_value, my_kmer)
    elif desc in ['PseDNC', 'PseKNC', 'PCPseDNC', 'PCPseTNC', 'SCPseDNC', 'SCPseTNC']:
        my_property_name, my_property_value, ok = CheckAccPseParameter.check_Pse_arguments(desc, descriptor.check_sequence_type(), desc_default_para)
        cmd = 'descriptor.' + desc + '(my_property_name, my_property_value)'
        status = eval(cmd)
    else:
        cmd = 'descriptor.' + desc + '()'
        status = eval(cmd)

    X = descriptor.encoding_array[1:][:,2:].astype(float)
    y = descriptor.encoding_array[1:][:,1].astype(int)
    groups = np.array(['_'.join(label.split('_')[:2]) for label in descriptor.encoding_array[1:][:,0]])

    return X,y,groups

def read_dataset(test_filename):
    """
    Reads the dataset from a FASTA file and extracts sequences, labels, groups, genera, and species.
    Args:
        test_filename (str): Path to the FASTA file.
    Returns:
        tuple: Contains numpy arrays of sequences, labels, groups, genera, and species.
    """
    X_test = []
    y_test = []
    groups_test = []
    genera_test = []
    species_test = []
    for record in SeqIO.parse(test_filename, "fasta"):
        X_test.append(str(record.seq))
        y_test.append(record.id.split('|')[1])
        groups_test.append(record.id.split('|')[0])
        # Extract genus and species from the description
        if '.' in record.description.split(',')[0].split(' ')[1]:
            idx = 2
        else:
            idx = 1
        genus = record.description.split(',')[0].split(' ')[idx]
        genus = genus.strip('[]')
        genera_test.append(genus)
        species_test.append(' '.join(record.description.split(',')[0].split(' ')[idx:]))

    X_test = np.array(X_test).astype(str)
    y_test = np.array(y_test).astype(int)
    groups_test = np.array(groups_test)
    genera_test = np.array(genera_test)
    species_test = np.array(species_test)
    return X_test, y_test, groups_test, genera_test, species_test

def create_fold(X, y, groups, n_splits, n_repeats):
    """
    Create stratified folds for cross-validation with groups.
    Parameters:
    ----------
    n_splits (int): Number of splits for cross-validation.
    n_repeats (int): Number of times to repeat the cross-validation.
    X (array-like): Feature matrix.
    y (array-like): Target values.
    groups (array-like): Group labels for stratification.
    Returns:
    -------
    all_runs (list): A list containing the folds for each repeat.
    """

    all_runs = []

    for repeat in range(n_repeats):
        # Shuffle inputs (preserving correspondence)
        rng = random.getrandbits(16)
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=rng)
        folds = []

        for fold_idx, (train_idx_full, test_idx) in enumerate(cv.split(X, y, groups)):
            # Split the data into train and test sets
            X_train, X_test = X[train_idx_full], X[test_idx]
            y_train, y_test = y[train_idx_full], y[test_idx]
            groups_train, groups_test = groups[train_idx_full], groups[test_idx]
            # split X_train to train and validation sets with groups stratified and take only 1 split
            cv_inner = StratifiedGroupKFold(n_splits=1, shuffle=True, random_state=rng)
            for inner_fold_idx, (inner_train_idx, inner_val_idx) in enumerate(cv_inner.split(X_train, y_train, groups_train)):
                if inner_fold_idx == 0:
                    train_idx = train_idx_full[inner_train_idx]
                    valid_idx = train_idx_full[inner_val_idx]
                    break
            folds.append({
                "fold": fold_idx,
                "train_idx": train_idx.tolist(),
                "valid_idx": valid_idx.tolist(),
                "test_idx": test_idx.tolist()
            })

        # Store the folds for this repeat
        all_runs.append({
            "repeat": repeat,
            "folds": folds
        })

    return all_runs

def read_fold(filename):
    # Load folds
    with open(filename, "r") as f:
        all_runs = json.load(f)

    n_repeats = len(all_runs)
    n_folds = len(all_runs[0]["folds"])

    # Create a matrix: (repeat * fold) x (repeat * fold)
    labels = {}
    valid_sets = []
    test_sets = []

    for repeat in range(n_repeats):
        for fold in range(n_folds):
            train_indices = all_runs[repeat]["folds"][fold]["train_idx"]
            val_indices = all_runs[repeat]["folds"][fold]["valid_idx"]
            test_indices = all_runs[repeat]["folds"][fold]["test_idx"]
            
            labels[f"fold_{(repeat*n_folds)+fold}"] = {'train_idx': train_indices, 'valid_idx': val_indices, 'test_idx': test_indices}

    return labels



# Fetching and saving FASTA files for the remaining genomes by ChatGPT

def batch_accessions(accessions, batch_size=3):
    """
    Yield successive n-sized chunks from accessions list.
    Parameters:
    ----------
    accessions (list): List of accession numbers.
    batch_size (int): Size of each batch.
    Returns:
    -------
    generator: Yields batches of accessions.
    """
    for i in range(0, len(accessions), batch_size):
        yield accessions[i:i + batch_size]

def fetch_and_save_individual_fastas(batch, destination_folder="dataset/genomes/islandviewer4/who_selected", silent=True):
    """
    Fetch and save FASTA files for a batch of accession numbers.
    Parameters:
    ----------
    batch (list): List of accession numbers.
    destination_folder (str): Folder to save the fetched FASTA files.
    Returns:
    -------
    None
    """

    ids = ",".join(batch)
    try:
        with Entrez.efetch(db="nucleotide", id=ids, rettype="fasta", retmode="text") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                # Ensure the destination folder exists
                os.makedirs(destination_folder, exist_ok=True)
                filename = os.path.join(destination_folder,f"{record.id}.fasta")
                if not os.path.exists(filename):
                    with open(filename, "w") as f:
                        SeqIO.write(record, f, "fasta")
                    if not silent:
                        print(f"Saved {filename}")
    except Exception as e:
        print(f"Error fetching batch {ids}: {e}")

########################## function to negative sampling ##########################
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
def sample_negative(genome, positives, neg_samples=0, neg_reference=f"{FOLDER_TRAIN}/islandpick_data.fasta"):
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
    islandviewer_gis : pandas.DataFrame, DataFrame containing positive regions from islandviewer4.
    genome_path : str, path to the directory containing genome fasta files.
    Returns:
    -------
    islandviewer_gis_df : pandas.DataFrame, DataFrame containing both positive and negative samples.
    """
    # # read samples genomes from islandviewer4
    # list_genomes = []
    # genomes_dict = {}

    # for file in os.listdir(genome_path):
    #     file_path = os.path.join(genome_path,file)
    #     if os.path.isfile(file_path) and file_path.endswith((".fasta",".fa",".fna")):
    #         list_genomes.append(file.split('.fasta')[0])
    #         seq_record = list(SeqIO.parse(file_path, "fasta"))[0]
    #         desc = seq_record.description
    #         sequence = str(seq_record.seq)
    #         genomes_dict.update({file.split('.fasta')[0]:{'Sequence':sequence,'Description':desc}}) 

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


    # # select sample genomes from islandviewer4 data
    # islandviewer_gis_sample = islandviewer_gis[islandviewer_gis['accession'].isin(genomes_dict.keys())].reset_index(drop=True)
    # islandviewer_gis_sample['Label'] = '1'

    # # collect positive regions per genome
    # genome = genomes_dict
    # positive_regions = islandviewer_gis_sample.groupby('accession').apply(lambda g: list(zip(g['start'], g['end']))).to_dict()

    # select sample genomes from positives data
    selected_ids = set(genomes_dict.keys()) & set(positives.accession.unique())
    pos_df = positives[positives['accession'].isin(selected_ids)].reset_index(drop=True)
    pos_df['Label'] = '1'

    # collect positive regions per genome
    genome = {k:genomes_dict[k] for k in selected_ids}
    positive_regions = pos_df.groupby('accession').apply(lambda g: list(zip(g['start'], g['end']))).to_dict()

    del genomes_dict

    negatives_samples = sample_negative(genome, positive_regions)

    # Convert to DataFrame
    neg_df = pd.DataFrame(negatives_samples, columns=["accession", "start", "end"]) 
    # select sample genomes from islandviewer4 data
    neg_df['Label'] = '0'
    neg_df['prediction_method'] = 'random_neg_sample'

    #combine pos and neg samples
    final_df = pd.concat([pos_df,neg_df])
    final_df['Sequence'] = final_df.apply(lambda x: genome[x['accession']]['Sequence'][x['start']:x['end']+1],axis=1)
    final_df['Description'] = final_df.apply(lambda x: f"{x['accession']}:{x['start']}-{x['end']} {genome[x['accession']]['Description']}", axis=1)
    final_df['gc'] = final_df.apply(lambda x: gc_fraction(x['Sequence']),axis=1)
    final_df.rename(columns={'accession':'Accession','start':'Start','end':'End'},inplace=True)

    return final_df
        

########################## function to read cross validation results ##########################
def read_results(filename, header=['dataset','model','fold','n_fold','representation']):
    """
    a function to read cross validation results

    Parameters: 
    ----------
    filename : str, file of cross validation results

    Returns: 
    -------
    cross_val_df : pandas.DataFrame
    """

    cols = pd.read_excel(filename, header=None,nrows=1).values[0]
    col_dict = {}
    eval_metrics = []

    for col in cols[len(header):]:
        if col not in col_dict.keys():
            col_dict.update({col:1})
        else:
            col_dict.update({col:col_dict[col]+1})

        eval_metrics.append(col+'_'+str(col_dict[col]))

    n_header_og = len(header)
    header.extend(eval_metrics)

    cross_val_df = pd.read_excel(filename, header=None, skiprows=1) # skip 1 row
    cross_val_df.columns = header
    #cross_val_df[header[0:n_header_og]] = cross_val_df[header[0:n_header_og]].fillna(method='ffill')
    cross_val_df[header[0:n_header_og]] = cross_val_df[header[0:n_header_og]].ffill()

    return cross_val_df

def compute_metrics(predictions_df, labels, y):
    """
    Compute evaluation metrics for each fold and return a DataFrame.
    
    Args:
        predictions_df (pd.DataFrame): DataFrame containing predictions.
        labels (dict): Dictionary containing train/validation/test indices for each fold.
        y (np.array): Ground truth labels.
        
    Returns:
        pd.DataFrame: DataFrame containing evaluation metrics for each fold.
    """
    eval_metrics = []

    for fold in labels.keys():
        fold_predictions_df = predictions_df[predictions_df['fold']==fold]
        fold_predictions_df = fold_predictions_df.dropna(axis=1)
        pred_cols = [col for col in fold_predictions_df.columns if col.startswith('y_')]
        df_preds = fold_predictions_df.set_index('pair_id')[pred_cols]

        test_indices = labels[fold]['valid_idx'] + labels[fold]['test_idx']
        ground_truth = y[test_indices]

        for pair_id, row in df_preds.iterrows():
            pred_prob = row.values
            pred_prob = pred_prob[:len(test_indices)]
            # Apply custom threshold
            threshold = 0.5
            preds = ((1-pred_prob) >= threshold).astype(int)
            mcc = matthews_corrcoef(ground_truth, preds)
            f1 = f1_score(ground_truth, preds)
            accuracy = accuracy_score(ground_truth, preds),
            precision = precision_score(ground_truth, preds),
            recall = recall_score(ground_truth, preds),
            true_positive = np.sum((ground_truth == 1) & (preds == 1))
            true_negative = np.sum((ground_truth == 0) & (preds == 0))
            false_positive = np.sum((ground_truth == 0) & (preds == 1))
            false_negative = np.sum((ground_truth == 1) & (preds == 0))

            # Append metrics for the current fold and pair_id
            eval_metrics.append({
                'fold': fold,
                'pair_id': pair_id,
                'mcc': mcc,
                'f1': f1,
                'precision': precision[0],
                'recall': recall[0],
                'accuracy': accuracy[0],
                'true_positive': true_positive,
                'true_negative': true_negative,
                'false_positive': false_positive,
                'false_negative': false_negative
            })
    
    return pd.DataFrame(eval_metrics)

def get_top_n_models(eval_metrics, n=5):
    """
    Get the top n models based on F1 score.
    
    Args:
        df (pd.DataFrame): DataFrame containing evaluation metrics.
        n (int): Number of top models to return.
        
    Returns:
        pd.DataFrame: DataFrame containing the top n models.
    """
    avg_eval_metrics = eval_metrics.groupby(['pair_id']).agg({'mcc':'mean', 'f1':'mean','precision':'mean','recall':'mean','accuracy':'mean'}).reset_index().sort_values(by=['f1'], ascending=False)
    avg_eval_metrics['representation'] = avg_eval_metrics['pair_id'].apply(lambda x: x.split('/')[0])
    avg_eval_metrics['model'] = avg_eval_metrics['pair_id'].apply(lambda x: x.split('/')[1])

    #rank the RCKmer and Kmer based on F1 score per dataset
    kmer_family = avg_eval_metrics[(avg_eval_metrics.representation.str.startswith('Kmer'))].copy()
    kmer_family['group_rank'] = kmer_family['f1'].rank(method="first", ascending=False)

    rckmer_family = avg_eval_metrics[(avg_eval_metrics.representation.str.startswith('RCKmer'))].copy()
    rckmer_family['group_rank'] = rckmer_family['f1'].rank(method="first", ascending=False)

    kmer_table = pd.concat([kmer_family,rckmer_family])

    #assign the rank of RCKmer and Kmer to the original table
    df_merged = pd.merge(avg_eval_metrics, kmer_table[['representation','group_rank']], on=['representation'], how='left')
    df_merged = df_merged.fillna(0.0)

    #select the best RCKmer and Kmer only
    sub_final_df = df_merged[df_merged['group_rank']<=1.0].copy()
    #select best model per representation
    sub_final_df['rank'] = sub_final_df.groupby(['representation'])['f1'].rank(method="first", ascending=False)
    sub_final_df = sub_final_df[sub_final_df['rank']==1]


    return sub_final_df.sort_values(by='f1', ascending=False).head(n)

########################## function to read boundary prediction evaluation results ##########################
def read_eval_result(json_file):
    """
    a function to read boundary prediction evaluation results

    Parameters: 
    ----------
    filename : str, file of boundary prediction evaluation results

    Returns: 
    -------
    eval_df : pandas.DataFrame
    """

    with open(json_file, 'r') as file:
        json_obj = json.load(file)
        eval_result = json.loads(json_obj)

    eval_df = pd.DataFrame()

    for predictor in eval_result.keys():
        eval = dict()

        for org in eval_result[predictor].keys():
            metrics_dict = dict()

            for metric in eval_result[predictor][org]:
                metrics_dict.update(metric)

            f_2_score = 0 if (metrics_dict['Precision'] == 0 or metrics_dict['Recall'] == 0) else (1 + 2**2) * (metrics_dict['Precision'] * metrics_dict['Recall']) / (2**2 * metrics_dict['Precision'] + metrics_dict['Recall'])
            metrics_dict.update({'F-2-Score':f_2_score})
            eval.update({org:metrics_dict})
        
        eval_df_temp = nested_dict_to_df(eval).reset_index()
        eval_df_temp = eval_df_temp.assign(Predictor=predictor)
        eval_df = pd.concat([eval_df,eval_df_temp])
    return eval_df 

########################## Define custom scorer ##########################
 
def specificity_score(y_true, y_pred):
    """
    a function to calculate specificity

    Parameters: 
    ----------
    y_true : array-like, ground truth
    y_pred : array_like, predictions

    Returns: 
    -------
    specificity : float
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    return specificity

def true_negative(y_true, y_pred):
    """
    a function to calculate true negative

    Parameters: 
    ----------
    y_true : array-like, ground truth
    y_pred : array_like, predictions

    Returns: 
    -------
    tn : int, number of true negative
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn

def true_positive(y_true, y_pred):
    """
    a function to calculate true positive

    Parameters: 
    ----------
    y_true : array-like, ground truth
    y_pred : array_like, predictions

    Returns: 
    -------
    tn : int, number of true positive
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp

def false_positive(y_true, y_pred):
    """
    a function to calculate false positive

    Parameters: 
    ----------
    y_true : array-like, ground truth
    y_pred : array_like, predictions

    Returns: 
    -------
    tn : int, number of false positive
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp

def false_negative(y_true, y_pred):
    """
    a function to calculate false negative

    Parameters: 
    ----------
    y_true : array-like, ground truth
    y_pred : array_like, predictions

    Returns: 
    -------
    fn : int, number of false negative
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fn

#define scoring metrics
scoring = {
    'Accuracy': 'accuracy',
    'Precision': 'precision',
    'Recall': 'recall',
    'Specificity': make_scorer(specificity_score),
    'F_1': 'f1',  # F-beta with beta=1 is equivalent to F1-score
    'F_beta_0.5': make_scorer(fbeta_score, beta=0.5),
    'F_beta_2': make_scorer(fbeta_score, beta=2),
    'MCC': make_scorer(matthews_corrcoef),
    'TP': make_scorer(true_positive),
    'TN': make_scorer(true_negative),
    'FP': make_scorer(false_positive),
    'FN': make_scorer(false_negative),
}


########################## function to evaluate model ##########################

def evaluate_model(train_file, test_file, train_params, **kwargs):
    """
    a function to evaluate model given train  data set and test data set

    Parameters: 
    ----------
    train_file : str, file for train data set
    test_file : str, file for test data set
    train_params : dict, parameters for training include data representation, model

    Returns: 
    -------
    model : trained sklearn estimator, the trained machine learning model.
    eval_scores : dict, evaluation results
    """
    if 'k' in train_params.keys():
        train_params['representation_params'].update({'kmer':train_params['k']})

    if train_params['representation'] in ['Kmer', 'RCKmer']:
        key = "{}-{}".format(train_params['representation'],train_params['representation_params']['kmer'])
    else:
        key = train_params['representation']

    print('representation Train Data with {}'.format(key))
    X_train,y_train,groups_train = get_representations(train_params['representation'], train_file, train_params['representation_params'])
    print('representation Test Data with {}'.format(key))
    X_test,y_test,groups_test = get_representations(train_params['representation'], test_file, train_params['representation_params'])

    groups_train = np.array([group.split('.')[0] for group in groups_train])
    groups_test = np.array([group.split('.')[0] for group in groups_test])

    # ensure training data does not contain test data
    test_ids = set(groups_test)

    selected_ids = []
    for i, g in enumerate(groups_train):
        if g not in test_ids:
            selected_ids.append(i)

    reduced_X_train = X_train[selected_ids]
    reduced_y_train = y_train[selected_ids]
    reduced_group_train = groups_train[selected_ids]

    #ensure species in train and test data sets are mutually exclusive
    assert len(set(groups_test)-set(reduced_group_train)) == len(set(groups_test))

    # #define the models to be trained
    # models = {
    #     'DecisionTree': DecisionTreeClassifier(random_state=42),
    #     'RandomForest': RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=10, min_samples_leaf=1, min_samples_split=2,max_features='sqrt',random_state=42),
    #     'LogisticRegression': LogisticRegression(max_iter=200, random_state=42),
    #     'SVM': SVC(random_state=42, kernel='rbf', C=2, gamma='scale', probability=True),
    #     'NaiveBayes': GaussianNB(),
    #     'XGBoost': XGBClassifier(n_estimators = 500,
    #                             use_label_encoder = False,
    #                             eval_metric = "logloss",
    #                             n_jobs = -1)
    # }

    #define the models to be trained
    models = {
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'NaiveBayes': GaussianNB(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'Bagging':BaggingClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42)
    }

    if 'model' in kwargs.keys():
        model = kwargs['model']
    elif 'model_path' in kwargs.keys():
        try:
            print('Loading model')
            with open(kwargs['model_path'], "rb") as input_file:
                model = pickle.load(input_file)
        except Exception as e:
            print(e)
    else:
        model = models[train_params['model']]

        print('Training in progress')
        model.fit(reduced_X_train,reduced_y_train)
        print('Training is done')

    print('Testing the model')
    y_pred = model.predict(X_test)
    #y_pred_prob = clf.predict_proba(X_test)[:,1]

    eval_scores = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F_1': f1_score(y_test, y_pred), 
        'F_beta_0.5': fbeta_score(y_test, y_pred, beta=0.5),
        'F_beta_2': fbeta_score(y_test, y_pred, beta=2),
        'MCC': matthews_corrcoef(y_test, y_pred),
        'TP': true_positive(y_test, y_pred),
        'TN': true_negative(y_test, y_pred),
        'FP': false_positive(y_test, y_pred),
        'FN': false_negative(y_test, y_pred),
    }

    if 'return_data' in kwargs.keys():
        return model, eval_scores, X_train, y_train, groups_train, X_test, y_test, groups_test, y_pred
    else:
        return model, eval_scores
    
def predict_output(train_params):
    """
    a function to evaluate model given train  data set and test data set

    Parameters: 
    ----------
    train_file : str, file for train data set
    test_file : str, file for test data set
    train_params : dict, parameters for training include data representation, model

    Returns: 
    -------
    model : trained sklearn estimator, the trained machine learning model.
    eval_scores : dict, evaluation results
    """

    train_file = train_params['train_file']
    test_file = train_params['test_file']
    
    if 'k' in train_params.keys():
        train_params['representation_params'].update({'kmer':train_params['k']})

    if train_params['representation'] in ['Kmer', 'RCKmer']:
        key = "{}-{}".format(train_params['representation'],train_params['representation_params']['kmer'])
    else:
        key = train_params['representation']

    logger.info('representation Train Data with {}'.format(key))
    X_train,y_train,groups_train = get_representations(train_file, train_params['representation'], train_params['representation_params'])
    logger.info('representation Test Data with {}'.format(key))
    X_test,y_test,groups_test = get_representations(test_file, train_params['representation'], train_params['representation_params'])

    groups_train = np.array([group.split('.')[0] for group in groups_train])
    groups_test = np.array([group.split('.')[0] for group in groups_test])

    # ensure training data does not contain test data
    test_ids = set(groups_test)

    selected_ids = []
    for i, g in enumerate(groups_train):
        if g not in test_ids:
            selected_ids.append(i)

    reduced_X_train = X_train[selected_ids]
    reduced_y_train = y_train[selected_ids]
    reduced_group_train = groups_train[selected_ids]

    #ensure species in train and test data sets are mutually exclusive
    assert len(set(groups_test)-set(reduced_group_train)) == len(set(groups_test))

    models = {
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'NaiveBayes': GaussianNB(),
        'GradientBoosting': GradientBoostingClassifier(),
        'Bagging':BaggingClassifier(),
        'XGBoost': XGBClassifier()
    }
    output = {}

    for model in models.keys():
        clf = models[model]
        logger.info('Training {} in progress - {}'.format(model, key))
        clf.fit(reduced_X_train,reduced_y_train)
        logger.info('Training {} is done - {}'.format(model, key))

        logger.info('Testing {} - {}'.format(model,key))
        y_pred = clf.predict(X_test)

        output[(key,model,'y')] = y_pred
    output[(key,'ground_truth','y')] = y_test

    return output

def predict_output_new(train_params):
    """
    a function to evaluate model given train  data set and test data set

    Parameters: 
    ----------
    train_file : str, file for train data set
    train_idx : list, indices for training samples
    test_idx : list, indices for test samples
    train_params : dict, parameters for training include data representation, model

    Returns: 
    -------
    model : trained sklearn estimator, the trained machine learning model.
    eval_scores : dict, evaluation results
    """

    train_file = train_params['train_file']
    folds = train_params['folds']
    
    if 'k' in train_params.keys():
        train_params['representation_params'].update({'kmer':train_params['k']})

    if train_params['representation'] in ['Kmer', 'RCKmer']:
        key = "{}-{}".format(train_params['representation'],train_params['representation_params']['kmer'])
    else:
        key = train_params['representation']

    logger.info('representation Train Data with {}'.format(key))
    X,y,groups = get_representations(train_file, train_params['representation'], train_params['representation_params'])

    n_repeats = len(folds)
    n_folds = len(folds[0]['folds'])
    train_indices = []
    val_indices = []
    test_indices = []
    output = {}

    for repeat in range(n_repeats):
        for fold in range(n_folds):
            train_indices.append(folds[repeat]['folds'][fold]['train_idx'])
            val_idx = folds[repeat]['folds'][fold]['valid_idx']
            test_idx = folds[repeat]['folds'][fold]['test_idx']
            test_indices.append(val_idx + test_idx)

    for fold,(train_idx, test_idx) in enumerate(zip(train_indices,test_indices)):  
        logger.info('Fold {}'.format(fold))  

        X_train, y_train, groups_train = X[train_idx], y[train_idx], groups[train_idx] 
        X_test, y_test, groups_test = X[test_idx], y[test_idx], groups[test_idx] 

        groups_train = np.array([group.split('.')[0] for group in groups_train])
        groups_test = np.array([group.split('.')[0] for group in groups_test])

        # ensure training data does not contain test data
        test_ids = set(groups_test)

        selected_ids = []
        for i, g in enumerate(groups_train):
            if g not in test_ids:
                selected_ids.append(i)

        reduced_X_train = X_train[selected_ids]
        reduced_y_train = y_train[selected_ids]
        reduced_group_train = groups_train[selected_ids]

        #ensure species in train and test data sets are mutually exclusive
        assert len(set(groups_test)-set(reduced_group_train)) == len(set(groups_test))

        models = {
            'DecisionTree': DecisionTreeClassifier(random_state=42),
            'RandomForest': RandomForestClassifier(random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'AdaBoost': AdaBoostClassifier(random_state=42),
            'NaiveBayes': GaussianNB(),
            'GradientBoosting': GradientBoostingClassifier(),
            'Bagging':BaggingClassifier(),
            'XGBoost': XGBClassifier()
        }
        
        for model in models.keys():
            clf = models[model]
            logger.info('Fold {}: Training {} in progress - {}'.format(fold,model, key))
            clf.fit(reduced_X_train,reduced_y_train)
            logger.info('Fold {}: Training {} is done - {}'.format(fold,model, key))

            logger.info('Fold {}: Testing {} - {}'.format(fold,model,key))
            y_pred = clf.predict_proba(X_test)
            class_0_prob = y_pred[:,0]
            output[(f'fold_{fold}',key,model,'y_0')] = np.round(class_0_prob,3)

    return output
    
def evaluate_ensemble_representation(train_file, test_file, train_params, **kwargs):
    """
    a function to evaluate model given train  data set and test data set

    Parameters: 
    ----------
    train_file : str, file for train data set
    test_file : str, file for test data set
    train_params : dict, parameters for training include data representation, model

    Returns: 
    -------
    model : trained sklearn estimator, the trained machine learning model.
    eval_scores : dict, evaluation results
    """
    if 'k' in train_params.keys():
        train_params['representation_params'].update({'kmer':train_params['k']})

    if train_params['representation_1'] in ['Kmer', 'RCKmer']:
        key_1 = "{}-{}".format(train_params['representation_1'],train_params['representation_params']['kmer'])
    else:
        key_1 = train_params['representation_1']

    if train_params['representation_2'] in ['Kmer', 'RCKmer']:
        key_2 = "{}-{}".format(train_params['representation_2'],train_params['representation_params']['kmer'])
    else:
        key_2 = train_params['representation_2']

    print('representation Train Data with {}'.format(key_1))
    X_train_1,y_train,groups_train = get_representations(train_params['representation_1'], train_file, train_params['representation_params'])
    print('representation Test Data with {}'.format(key_1))
    X_test_1,y_test,groups_test = get_representations(train_params['representation_1'], test_file, train_params['representation_params'])

    print('representation Train Data with {}'.format(key_2))
    X_train_2,y_train,groups_train = get_representations(train_params['representation_2'], train_file, train_params['representation_params'])
    print('representation Test Data with {}'.format(key_2))
    X_test_2,y_test,groups_test = get_representations(train_params['representation_2'], test_file, train_params['representation_params'])

    # Concatenate the two feature sets
    X_train = np.concatenate((X_train_1, X_train_2), axis=1)
    X_test = np.concatenate((X_test_1, X_test_2), axis=1)
    
    # Convert groups to numpy arrays
    groups_train = np.array([group.split('.')[0] for group in groups_train])
    groups_test = np.array([group.split('.')[0] for group in groups_test])

    # ensure training data does not contain test data
    test_ids = set(groups_test)

    selected_ids = []
    for i, g in enumerate(groups_train):
        if g not in test_ids:
            selected_ids.append(i)

    reduced_X_train = X_train[selected_ids]
    reduced_y_train = y_train[selected_ids]
    reduced_group_train = groups_train[selected_ids]

    #ensure species in train and test data sets are mutually exclusive
    assert len(set(groups_test)-set(reduced_group_train)) == len(set(groups_test))

    #define the models to be trained
    models = {
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=10, min_samples_leaf=1, min_samples_split=2,max_features='sqrt',random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=200, random_state=42),
        'SVM': SVC(random_state=42, kernel='rbf', C=2, gamma='scale', probability=True),
        'NaiveBayes': GaussianNB(),
    }

    if 'model' in kwargs.keys():
        model = kwargs['model']
    elif 'model_path' in kwargs.keys():
        try:
            print('Loading model')
            with open(kwargs['model_path'], "rb") as input_file:
                model = pickle.load(input_file)
        except Exception as e:
            print(e)
    else:
        model = models[train_params['model']]

        print('Training in progress')
        model.fit(reduced_X_train,reduced_y_train)
        print('Training is done')

    print('Testing the model')
    y_pred = model.predict(X_test)
    #y_pred_prob = clf.predict_proba(X_test)[:,1]

    eval_scores = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F_1': f1_score(y_test, y_pred), 
        'F_beta_0.5': fbeta_score(y_test, y_pred, beta=0.5),
        'F_beta_2': fbeta_score(y_test, y_pred, beta=2),
        'MCC': matthews_corrcoef(y_test, y_pred),
        'TP': true_positive(y_test, y_pred),
        'TN': true_negative(y_test, y_pred),
        'FP': false_positive(y_test, y_pred),
        'FN': false_negative(y_test, y_pred),
    }

    if 'return_data' in kwargs.keys():
        return model, eval_scores, X_train, y_train, groups_train, X_test, y_test, groups_test
    else:
        return model, eval_scores

#####################################################################

def read_file(fasta_file, label):
    """
    a function to a read fasta file and transform it to a dataframe

    Parameters: 
    ----------
    fasta_file : str, fasta file to read
    label : str, label to assign to the data ('1' or '0')

    Returns: 
    -------
    output_df : pandas.DataFrame
    """

    # Read the FASTA file
    sequences = SeqIO.parse(fasta_file, "fasta")
    list_id = []
    position = ""
    # Iterate over each sequence in the FASTA file
    for seq_record in sequences:
        #print(f"ID: {seq_record.id}")
        if '..' in seq_record.id:
            position = re.search("\d+\..\d+", seq_record.id)[0]
        elif '..' in seq_record.description:
            position = re.search("\d+\..\d+", seq_record.description)[0]
        else:
            position = ""
    
        start = 0
        end = 0
        
        if position != "":
            start = position.split('..')[0]
            end = position.split('..')[1]
    
        id_ = '_'.join(seq_record.id.split('_')[:2])
        list_id.append((id_, label, int(start), int(end)))
    
    output_df = pd.DataFrame(list_id, columns = ['Accession','Label','Start','End'])
    
    return output_df

#query sequence from reference database
def query_sequence(accession_id, start=0, end=0):
    """
    a function to a query sequence from genomic database

    Parameters: 
    ----------
    accession_id : str, accession id of the genome of interes
    start : int, start position of the genome
    end : int, end position of the genome

    Returns: 
    -------
    [sequence, sequence's description]
    """

    try:
        if start==0 & end==0:
            handle = Entrez.efetch(db='nucleotide',
                               id=accession_id, 
                               rettype="fasta")
        else:
            handle = Entrez.efetch(db='nucleotide',
                                   id=accession_id, 
                                   rettype="fasta",
                                   seq_start=start,
                                   seq_stop=end)
            
        seq = SeqIO.read(handle, "fasta")
        handle.close()
    
        return [str(seq.seq), seq.description]
    except Exception as e:
        print(f"An error occurred: {e}")
        return ["retrieval failed", "retrieval failed"]
    
#write dataframe to fasta file

def df_to_fasta(data, dna_only=True, query_db=False, **kwargs):
    """
    a lambda function to process each row of a dataframe and transform it to a fasta file

    Parameters: 
    ----------
    data : a row of a pandas.DataFrame with columns ['Accession','Start','End','Label']
    or ['Accession','Start','End','Label','Sequence','Description']
    dna_only : bool, whether to return dna sequence only or include IUPAC code
    query_db : bool, whether or not query

    Returns: 
    -------
    result : str, a record in a fasta file ">accession|label|description\nsequence"
    """

    accession = data['Accession']
    label = data['Label']
    result = ""

    if query_db:
        seq_start = data['Start']
        seq_end = data['End']
        query = query_sequence(accession, seq_start, seq_end)
        sequence, description = query[0], query[1]
    else:
        sequence = data['Sequence']
        description = data['Description']

    try:
        if label.isdigit():
            if int(label) == 1:
                identifier = "GI_{}".format(int(data['rank']))
                label = 1
            else:
                identifier = "Non_GI_{}".format(int(data['rank']))
                label = 0
    
        else:
            if label != "negative":
                identifier = "GI_{}".format(int(data['rank']))
                label = 1
            else:
                identifier = "Non_GI_{}".format(int(data['rank']))
                label = 0
    except:
        identifier = ""
        if label.isdigit():
            if int(label) == 1:
                label = 1
            else:
                label = 0
    
        else:
            if label != "negative":
                label = 1
            else:
                label = 0

    if dna_only:
        if len(set(sequence) - set({'A','T','G','C'})) == 0:
            sequence = replace_iupac_with_nucleotide(sequence)

    if identifier != "":
        result = ">{}_{}|{}|{}\n{}".format(accession,identifier,label,description,sequence)
    else:
        result = ">{}|{}|{}\n{}".format(accession,label,description,sequence)
    
    try:
        if kwargs['write_file']:
            destination_file = kwargs['filename']
            # Writing sequences to a FASTA file
            with open(destination_file, 'a') as f:
                f.write(result + '\n')
    except Exception as e:
        return result

def check_sequence_type(fasta_list):
        """
        Specify sequence type (Protein, DNA or RNA)
        :return: str, type of the given sequence
        """
        if type(fasta_list) == str:
            fasta_list = [fasta_list]
        
        tmp_fasta_list = []
        if len(fasta_list) < 100:
            tmp_fasta_list = fasta_list
        else:
            random_index = random.sample(range(0, len(fasta_list)), 100)
            for i in random_index:
                tmp_fasta_list.append(fasta_list[i])

        sequence = ''
        for item in tmp_fasta_list:
            sequence += item

        char_set = set(sequence)
        if 5 < len(char_set) <= 21:
            for line in fasta_list:
                line = re.sub('[^ACDEFGHIKLMNPQRSTVWY]', '-', line)
            return 'Protein'
        elif 0 < len(char_set) <= 5 and 'T' in char_set:
            return 'DNA'
        elif 0 < len(char_set) <= 5 and 'U' in char_set:
            for line in fasta_list:
                line = re.sub('U', 'T', line)
            return 'RNA'
        else:
            return 'Unknown'
            
def fasta_to_df(file, dna_only=True):
    """
    a function to read fasta file and convert it into a pandas.DataFrame

    Parameters: 
    ----------
    file : str, fasta file
    dna_only : bool, whether or not to return only records of dna sequences 

    Returns: 
    -------
    output_df : pandas.DataFrame, columns ['Accession','Sequence','Start','End','Description','Label']
    """

    sequences = SeqIO.parse(file, "fasta")
    
    data = []
    
    # Iterate over each sequence in the FASTA file
    for seq_record in sequences:
        desc = seq_record.description
        #accession = '_'.join(desc.split('|')[0].split('_')[:2])
        accession = seq_record.id.split('|')[0]
        label = seq_record.id.split('|')[1]
        position = re.search("\d+\-\d+", seq_record.id)[0]
        start = int(position.split('-')[0])
        end = int(position.split('-')[1])
        sequence = str(seq_record.seq)
    
        if dna_only:
            if len(set(sequence) - set({'A','T','G','C'})) == 0:
                data.append((accession, sequence, start, end, desc, label))
            else:
                new_sequence = replace_iupac_with_nucleotide(sequence)
                data.append((accession, new_sequence, start, end, desc, label))
        else:
            data.append((accession, sequence, start, end, desc, label))
            
    output_df = pd.DataFrame(data, columns = ['Accession','Sequence','Start','End','Description','Label'])

    return output_df

def replace_iupac_with_nucleotide(sequence):
    """
    a function to replace IUPAC codes in a dna sequence with DNA letters

    Parameters: 
    ----------
    sequence : str, dna sequence 

    Returns: 
    -------
    str, original sequence with replaced IUPAC codes
    """

    original_sequence = []

    # Define the mapping from IUPAC codes to possible nucleotides
    iupac_map = {
        'A': ['A'],       # Adenine
        'C': ['C'],       # Cytosine
        'G': ['G'],       # Guanine
        'T': ['T'],       # Thymine
        'R': ['A', 'G'],  # Purine
        'Y': ['C', 'T'],  # Pyrimidine
        'S': ['G', 'C'],  # Strong
        'W': ['A', 'T'],  # Weak
        'K': ['G', 'T'],  # Keto
        'M': ['A', 'C'],  # Amino
        'B': ['C', 'G', 'T'],  # Not A
        'D': ['A', 'G', 'T'],  # Not C
        'H': ['A', 'C', 'T'],  # Not G
        'V': ['A', 'C', 'G'],  # Not T
        'N': ['A', 'C', 'G', 'T']  # Any nucleotide
    }
    
    for char in sequence:
        if char in iupac_map:
            # Choose one of the possible nucleotides at random
            chosen_nucleotide = random.choice(iupac_map[char])
            original_sequence.append(chosen_nucleotide)
        else:
            original_sequence.append(char)  # For standard nucleotides A, C, G, T
    return ''.join(original_sequence)

def get_organism_info(data_set):
    """
    a function to get organism info from a pandas.DataFrame

    Parameters: 
    ----------
    data_set : pandas.DataFrame, column = ['Accesion','Start','End']

    Returns: 
    -------
    organism : dict, dictionary of organisms with format {Accession:[Accession, Start, End]}
    """
    organism = {}
    for i, data in data_set.iterrows():
        acc = data.Accession
        if acc in organism.keys():
            organism[acc].append([acc, data.Start, data.End])
        else:
            organism[acc] = [[acc, data.Start, data.End]]
    return organism

def adjusted_rv(X, Y):
    """
    Calculate the adjusted RV coefficient between two datasets.
    Parameters:
    ----------
    X : array-like, shape (n_samples, n_features)
        First dataset.
    Y : array-like, shape (n_samples, n_features)
        Second dataset.
    Returns:
    -------
    adjusted_rv : float
        Adjusted RV coefficient between the two datasets.
    """
    # Center matrices
    X_centered = X - np.mean(X, axis=0)
    Y_centered = Y - np.mean(Y, axis=0)

     # Center and scale both datasets
    #X_centered = StandardScaler().fit_transform(X)
    #Y_centered = StandardScaler().fit_transform(Y)
    
    # Compute cross-product matrices
    Wx = X_centered @ X_centered.T
    Wy = Y_centered @ Y_centered.T
    
    # Standard RV coefficient
    numerator = np.trace(Wx @ Wy)
    denominator = np.sqrt(np.trace(Wx @ Wx) * np.trace(Wy @ Wy))
    rv = numerator / denominator
    
    # Expected RV under independence
    n = X.shape[0]
    trace_Wx = np.trace(Wx)
    trace_Wy = np.trace(Wy)
    trace_Wx2 = np.trace(Wx @ Wx)
    trace_Wy2 = np.trace(Wy @ Wy)
    
    E_rv = (trace_Wx * trace_Wy) / ((n - 1)**2 * np.sqrt(trace_Wx2 * trace_Wy2))
    
    # Adjusted RV
    adjusted_rv = (rv - E_rv) / (1 - E_rv)
    return adjusted_rv
    

def accession_to_taxid(acc):
    """Get TaxID for a given accession."""
    
    handle = Entrez.esearch(db="nucleotide", term=acc)
    record = Entrez.read(handle)
    handle.close()
    if record["IdList"]:
        nuccore_id = record["IdList"][0]
        handle = Entrez.esummary(db="nucleotide", id=nuccore_id)
        summary = Entrez.read(handle)
        handle.close()
        return summary[0].get("TaxId")
    return None

def taxid_to_lineage(taxid):
    """Get lineage information for a given TaxID."""
  
    handle = Entrez.efetch(db="taxonomy", id=taxid)
    records = Entrez.read(handle)
    handle.close()
    if records:
        lineage = {}
        for taxon in records[0].get("LineageEx", []):
            rank = taxon.get("Rank")
            name = taxon.get("ScientificName")
            if rank in ["family", "genus"]:
                lineage[rank] = name
        # Add species
        lineage["species"] = records[0]["ScientificName"]
        return lineage
    return {}

def accession_to_taxonomy(acc):
    """Return a dict with genus, species, family for an accession."""
    taxid = accession_to_taxid(acc)
    if taxid:
        return taxid_to_lineage(taxid)
    else:
        return {}
    
# Define a function to apply per group
def get_taxonomy(df):
    acc = df['accession'].values

    try:
        info = accession_to_taxonomy(acc)

        return pd.Series({
            'Genus': info.get('genus'),
            'Species': info.get('species'),
            'Family': info.get('family'),
        })
    except Exception as e:
        return pd.Series({'Genus': None, 'Species': None, 'Family': None})


# accessions = selected_accession_who  # Add your list here
# remaining_taxonomy_info = {}
# remaining_accessions = set(accessions)  - set(available_acc)# Use a set for faster lookups
# meta_df: one row per species
# columns: ['species', 'genus', 'n_samples', ...]

def sample_n_subtaxa_per_taxa(meta_df, taxa='genus', n_subtaxa=1, min_pos_per_subtaxa=1, random_state=42):
    """
    Sample 1 species per genus, optionally requiring a minimum number of positives.
    """
    # Optionally filter by minimum positives
    df = meta_df[meta_df['n_samples'] >= min_pos_per_subtaxa].copy()

    # Group by genus and sample n species from each
    sampled = (
        df.groupby(taxa, group_keys=True)
          .apply(lambda g: g if len(g) <= n_subtaxa else g.sample(n=n_subtaxa, random_state=random_state),
                 include_groups=False)
          .reset_index()
          .drop(columns='level_1')
    )
    return sampled

def sample_n_species_per_genus(meta_df, n_species=1, min_pos_per_species=1, random_state=42):
    """
    Sample 1 species per genus, optionally requiring a minimum number of positives.
    """
    # Optionally filter by minimum positives
    df = meta_df[meta_df['n_samples'] >= min_pos_per_species].copy()

    # Group by genus and sample n species from each
    sampled = (
        df.groupby('genus', group_keys=True)
          .apply(lambda g: g if len(g) <= n_species else g.sample(n=n_species, random_state=random_state),
                 include_groups=False)
          .reset_index()
          .drop(columns='level_1')
    )
    return sampled

def sample_n_strains_per_species(meta_df, n_strains=1, min_pos_per_species=1, random_state=42):
    """
    Sample 1 species per genus, optionally requiring a minimum number of positives.
    """
    # Optionally filter by minimum positives
    df = meta_df[meta_df['n_samples'] >= min_pos_per_species].copy()

    # Group by genus and sample n species from each
    sampled = (
        df.groupby('species', group_keys=True)
          .apply(lambda g: g if len(g) <= n_strains else g.sample(n=n_strains, random_state=random_state),
                 include_groups=False)
          .reset_index()
          .drop(columns='level_1')
    )
    return sampled

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

# import plotly.graph_objects as go

# def build_taxonomy_sankey(df, title, levels=None, weight_col=None):
#     """
#     Build a Sankey diagram across taxonomic levels.

#     df: DataFrame with columns for each taxonomic level
#     levels: ordered list of column names, e.g.
#             ['species', 'genus', 'family', 'order', 'phylum']
#     weight_col: column to use as weight (e.g., 'n_positive' or None = count rows)
#     """
#     if levels is None:
#         levels = ['species', 'genus', 'family', 'order', 'phylum']

#     # if no explicit weight column, use 1 per row
#     if weight_col is None:
#         df = df.copy()
#         df['_weight_'] = 1
#         weight_col = '_weight_'

#     # build all node labels
#     labels = []
#     for lvl in levels:
#         labels.extend(df[lvl].unique().tolist())
#     labels = list(dict.fromkeys(labels))  # unique while preserving order

#     # mapping label -> index
#     label_to_idx = {lab: i for i, lab in enumerate(labels)}

#     sources = []
#     targets = []
#     values = []

#     # build links between consecutive levels
#     for i in range(len(levels) - 1):
#         src_lvl = levels[i]
#         tgt_lvl = levels[i + 1]

#         grouped = (
#             df.groupby([src_lvl, tgt_lvl])[weight_col]
#               .sum()
#               .reset_index()
#         )

#         for _, row in grouped.iterrows():
#             src_label = row[src_lvl]
#             tgt_label = row[tgt_lvl]
#             val = row[weight_col]

#             sources.append(label_to_idx[src_label])
#             targets.append(label_to_idx[tgt_label])
#             values.append(val)

#     fig = go.Figure(data=[go.Sankey(
#         arrangement="snap",
#         node=dict(
#             pad=10,
#             thickness=15,
#             line=dict(color="black", width=0.3),
#             label=labels,
#         ),
#         link=dict(
#             source=sources,
#             target=targets,
#             value=values,
#         )
#     )])

#     fig.update_layout(
#         title_text=title,
#         font=dict(size=10),
#         width=1000, height=1000
#     )
#     return fig

def df_to_fasta_new(data, dna_only=True, query_db=False, **kwargs):
    """
    a lambda function to process each row of a dataframe and transform it to a fasta file

    Parameters: 
    ----------
    data : a row of a pandas.DataFrame with columns ['Accession','Start','End','Label']
    or ['Accession','Start','End','Label','Sequence','Description']
    dna_only : bool, whether to return dna sequence only or include IUPAC code
    query_db : bool, whether or not query

    Returns: 
    -------
    result : str, a record in a fasta file ">accession|label|description\nsequence"
    """

    accession = data['Accession']
    label = data['Label']
    result = ""

    if query_db:
        seq_start = data['Start']
        seq_end = data['End']
        query = query_sequence(accession, seq_start, seq_end)
        sequence, description = query[0], query[1]
    else:
        sequence = data['Sequence']
        description = data['Description']

    if dna_only:
        if len(set(sequence) - set({'A','T','G','C'})) == 0:
            sequence = replace_iupac_with_nucleotide(sequence)

    result = ">{}|{}|{}\n{}".format(accession,label,description,sequence)
    
    try:
        if kwargs['write_file']:
            destination_file = kwargs['filename']
            # Writing sequences to a FASTA file
            with open(destination_file, 'a') as f:
                f.write(result + '\n')
    except Exception as e:
        return result


# from sklearn.model_selection import GridSearchCV, PredefinedSplit
# from sklearn.metrics import accuracy_score

# results = []

# for fold_name, indices in folds_dict.items():
#     # 1. Extract indices for this specific fold
#     train_idx = indices['train_idx']
#     valid_idx = indices['valid_idx']
#     test_idx  = indices['test_idx']
    
#     # 2. Combine Train and Valid for the GridSearchCV
#     # We need a 'test_fold' array where Train = -1 and Valid = 0
#     X_tuning = X.iloc[np.concatenate([train_idx, valid_idx])]
#     y_tuning = y.iloc[np.concatenate([train_idx, valid_idx])]
    
#     # Create the predefined split indicator
#     # -1 for all training rows, 0 for all validation rows
#     test_fold = np.concatenate([
#         -1 * np.ones(len(train_idx)), 
#          0 * np.ones(len(valid_idx))
#     ])
#     ps = PredefinedSplit(test_fold)

#     # 3. Perform Hyperparameter Tuning (Inner Loop)
#     grid_search = GridSearchCV(model, param_grid, cv=ps)
#     grid_search.fit(X_tuning, y_tuning)
    
#     # 4. Evaluate the best model on the true Test Set (Outer Loop)
#     best_model = grid_search.best_estimator_
#     test_preds = best_model.predict(X.iloc[test_idx])
#     final_acc = accuracy_score(y.iloc[test_idx], test_preds)
    
#     results.append({
#         'fold': fold_name,
#         'best_params': grid_search.best_params_,
#         'test_accuracy': final_acc
#     })

# # Convert to DataFrame to see your 5-fold generalization stability
# results_df = pd.DataFrame(results)



def extract_all_functional_zones(accession, start, end):
    handle = Entrez.efetch(
        db="nucleotide", id=accession, seq_start=start, seq_stop=end,
        rettype="gbwithparts", retmode="text"
    )
    record = SeqIO.read(handle, "genbank")
    handle.close()
    
    features_list = []
    
    for feature in record.features:
        if feature.type in ["CDS"]:
            product = "".join(feature.qualifiers.get("product", [])).lower()
            note = "".join(feature.qualifiers.get("note", [])).lower()
            gene = "".join(feature.qualifiers.get("gene", [])).lower()
            
            # Combine text fields to scan them all at once
            all_text = f"{product} {note} {gene}"
            
            # Define classification rules
            if any(k in all_text for k in ["transposase", "integrase", "recombinase", "insertion sequence", "phage", "plasmid","IS"]):
                category = "Mobility"
            elif any(k in all_text for k in ["resistance", "beta-lactamase", "carbapenem","mercury", "arsenic", "biocide", "qac", "efflux","multidrug","drug","dmt","pace","mop","rnd","abc","mfs"]):
                category = "AMR"
            #elif any(k in all_text for k in ["ribosomal protein", "dna polymerase", "gyrase", "rna polymerase", "tnaA"]):
            #    category = "Housekeeping (Core)"
            else:
                category = "Other Functionality"
                
            global_start = int(feature.location.start) + start
            global_end = int(feature.location.end) + start
            
            features_list.append({
                "Gene_Symbol": feature.qualifiers.get("gene", [feature.type])[0],
                "Category": category,
                "Accession": accession,
                "Start": start,
                "End": end,
                "Region_Start": global_start,
                "Region_End": global_end,
                "Product_Description": feature.qualifiers.get("product", [""])[0],
                "Note": feature.qualifiers.get("note", [""])[0]
            })
            
    return pd.DataFrame(features_list)