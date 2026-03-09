# import libraries
from tqdm import tqdm
import pandas as pd 
from pandarallel import pandarallel
import os, sys
import pickle
from pathlib import Path
import json
from utils.data_processing import (fasta_to_df, df_to_fasta)
from utils.training import read_dataset,create_fold

tqdm.pandas()
# Initialize pandarallel
pandarallel.initialize(progress_bar=True)

file_path = os.path.split(os.path.realpath(__file__))[0]
pPath = Path(file_path).parent
sys.path.append(pPath)


FOLDER_TRAIN_DATA = "./dataset/train_folder"
FOLDER_TEST_DATA = "./dataset/test_folder"
TRAINING_POOL_FILENAME = "deduplicated_training_pool.fasta"
# define prefix for saving the training data
DATA_PREFIX = "deduplicated" #train, deduplicated, etc.

if __name__ == "__main__":

    print("====================Reading Training Pool FASTA file====================")
    # write the training data pool
    training_pool_file = f"{FOLDER_TRAIN_DATA}/training_pool/{TRAINING_POOL_FILENAME}"
    training_pool_df = fasta_to_df(training_pool_file)

    os.makedirs(f"{FOLDER_TRAIN_DATA}/data",exist_ok=True)

    print("====================Writing Training Case1====================")
    #Case 1: vary species and samples

    with open(f"{FOLDER_TRAIN_DATA}/info/{DATA_PREFIX}_case1.pkl", "rb") as f:
        selected_species_dict = pickle.load(f)

    n_species_per_genus = selected_species_dict.keys()

    for n in n_species_per_genus:
        selected_species = selected_species_dict[n].accession_version.tolist()
        pool_data_df_filtered = training_pool_df[training_pool_df['Accession'].isin(selected_species)]
        
        filename = f"{FOLDER_TRAIN_DATA}/data/{DATA_PREFIX}_case1_{n}.fasta"

        if not os.path.exists(filename):
            # write df to fasta file
            param = {'write_file':True,'filename':filename} #whether write sequences to a fasta file specified in filename
            fasta_data = pool_data_df_filtered.progress_apply(lambda x: df_to_fasta(x,
                                                                        dna_only=True, #dna_only determines whether or not to process sequences with IUPAC codes
                                                                        query_db=False, #query sequence from database
                                                                        **param), axis=1)
    

    print("====================Writing Training Case2a====================")
    #Case 2a: balanced samples between datasets with proportional ratio to the original samples

    with open(f"{FOLDER_TRAIN_DATA}/info/{DATA_PREFIX}_case2a.pkl", "rb") as f:
        selected_species_dict = pickle.load(f)

    n_species_per_genus = selected_species_dict.keys()

    for n in n_species_per_genus:
        selected_species = selected_species_dict[n].accession_version.tolist()
        sampled = training_pool_df[training_pool_df['Accession'].isin(selected_species)]
        # for each accession, sample up to cap number of samples per label
        cap_dict = selected_species_dict[n][['accession_version','cap']].set_index('accession_version').to_dict()['cap']

        pool_data_df_filtered = pd.DataFrame()
        for accession in cap_dict.keys():
            n_samples = cap_dict[accession]
            #positive samples
            pos_samples = sampled[(sampled['Accession']==accession) & (sampled['Label']=='1')].apply(lambda x: x.sample(n=min(len(x),n_samples), random_state=0))
            #negative samples
            neg_samples = sampled[(sampled['Accession']==accession) & (sampled['Label']=='0')].apply(lambda x: x.sample(n=min(len(x),n_samples), random_state=0))
            pool_data_df_filtered = pd.concat([pool_data_df_filtered, pos_samples, neg_samples], axis=0)

        filename = f"{FOLDER_TRAIN_DATA}/data/{DATA_PREFIX}_case2a_{n}.fasta"

        if not os.path.exists(filename):
            # write df to fasta file
            param = {'write_file':True,'filename':filename} #whether write sequences to a fasta file specified in filename
            fasta_data = pool_data_df_filtered.progress_apply(lambda x: df_to_fasta(x,
                                                                        dna_only=True, #dna_only determines whether or not to process sequences with IUPAC codes
                                                                        query_db=False, #query sequence from database
                                                                        **param), axis=1)
        
    print("====================Writing Training Case2b====================")
    #Case 2b: balanced samples between datasets with balanced ratio between species

    with open(f"{FOLDER_TRAIN_DATA}/info/{DATA_PREFIX}_case2b.pkl", "rb") as f:
        selected_species_dict = pickle.load(f)

    n_species_per_genus = selected_species_dict.keys()

    for n in n_species_per_genus:
        selected_species = selected_species_dict[n].accession_version.tolist()
        sampled = training_pool_df[training_pool_df['Accession'].isin(selected_species)]
        # for each accession, sample up to cap number of samples per label
        cap_dict = selected_species_dict[n][['accession_version','cap']].set_index('accession_version').to_dict()['cap']

        pool_data_df_filtered = pd.DataFrame()
        for accession in cap_dict.keys():
            n_samples = cap_dict[accession]
            #positive samples
            pos_samples = sampled[(sampled['Accession']==accession) & (sampled['Label']=='1')].apply(lambda x: x.sample(n=min(len(x),n_samples), random_state=0))
            #negative samples
            neg_samples = sampled[(sampled['Accession']==accession) & (sampled['Label']=='0')].apply(lambda x: x.sample(n=min(len(x),n_samples), random_state=0))
            pool_data_df_filtered = pd.concat([pool_data_df_filtered, pos_samples, neg_samples], axis=0)

        filename = f"{FOLDER_TRAIN_DATA}/data/{DATA_PREFIX}_case2b_{n}.fasta"

        if not os.path.exists(filename):
            # write df to fasta file
            param = {'write_file':True,'filename':filename} #whether write sequences to a fasta file specified in filename
            fasta_data = pool_data_df_filtered.progress_apply(lambda x: df_to_fasta(x,
                                                                        dna_only=True, #dna_only determines whether or not to process sequences with IUPAC codes
                                                                        query_db=False, #query sequence from database
                                                                        **param), axis=1).to_list()
        

    print("====================Writing Training Case3====================")
    #Case 3: fixed species, vary samples

    with open(f"{FOLDER_TRAIN_DATA}/info/{DATA_PREFIX}_case3.pkl", "rb") as f:
        selected_species_dict = pickle.load(f)

    n_species_per_genus = selected_species_dict.keys()

    for n in n_species_per_genus:
        selected_species = selected_species_dict[n].accession_version.tolist()
        sampled = training_pool_df[training_pool_df['Accession'].isin(selected_species)]
        # for each accession, sample up to cap number of samples per label
        cap_dict = selected_species_dict[n][['accession_version','cap']].set_index('accession_version').to_dict()['cap']

        pool_data_df_filtered = pd.DataFrame()
        for accession in cap_dict.keys():
            n_samples = cap_dict[accession]
            #positive samples
            pos_samples = sampled[(sampled['Accession']==accession) & (sampled['Label']=='1')].apply(lambda x: x.sample(n=min(len(x),n_samples), random_state=0))
            #negative samples
            neg_samples = sampled[(sampled['Accession']==accession) & (sampled['Label']=='0')].apply(lambda x: x.sample(n=min(len(x),n_samples), random_state=0))
            pool_data_df_filtered = pd.concat([pool_data_df_filtered, pos_samples, neg_samples], axis=0)

        filename = f"{FOLDER_TRAIN_DATA}/data/{DATA_PREFIX}_case3_{n}.fasta"

        if not os.path.exists(filename):
            # write df to fasta file
            param = {'write_file':True,'filename': filename} #whether write sequences to a fasta file specified in filename
            fasta_data = pool_data_df_filtered.progress_apply(lambda x: df_to_fasta(x,
                                                                        dna_only=True, #dna_only determines whether or not to process sequences with IUPAC codes
                                                                        query_db=False, #query sequence from database
                                                                        **param), axis=1).to_list()
    

    print("====================Create pre-defined folds for the training====================")
    # create pre-defined folds for the training
    train_cases = ['case1', 'case2a', 'case2b', 'case3']
    n_subcases = 5

    for train_case in train_cases:
        for n in range(1,n_subcases+1):
            print(f"{train_case}_{n}")

            folds_folder = f"{FOLDER_TRAIN_DATA}/folds"

            os.makedirs(folds_folder, exist_ok=True)

            folds_file = f"{folds_folder}/{DATA_PREFIX}_{train_case}_{n}_folds.json"

            if not os.path.exists(folds_file):
                # read data
                filename = f"{FOLDER_TRAIN_DATA}/data/{DATA_PREFIX}_{train_case}_{n}.fasta" 
              
                X, y, groups, genera, species = read_dataset(filename)
          
                n_splits = 5
                n_repeats = 5
                all_runs = create_fold(X, y, species, n_splits=n_splits, n_repeats=n_repeats)
                
                
                # Save to file
                with open(folds_file, "w") as f:
                    json.dump(all_runs, f, indent=2)
            else:
                print(f"{folds_file} already exists!")


