# import libraries
import wget
import tarfile 
import gzip
from tqdm import tqdm
import pandas as pd 
import sqlite3
from pandarallel import pandarallel
import os, sys
import pickle
from pathlib import Path
from utils.data_processing import (accession_to_taxid,
                                   map_taxon_to_df,
                                   sample_genus_species,
                                   sample_n_subtaxa_per_taxa,
                                   batch_accessions,
                                   fetch_and_save_individual_fastas,
                                   fasta_to_df,
                                   df_to_fasta)
from utils.data_sampling import proportional_caps,uniform_caps,negatives_sampling
#from utils.training import read_dataset,create_fold

tqdm.pandas()
# Initialize pandarallel
pandarallel.initialize(progress_bar=True)

file_path = os.path.split(os.path.realpath(__file__))[0]
pPath = Path(file_path).parent
sys.path.append(pPath)


FOLDER_TRAIN_DATA = "./dataset/train_folder"
FOLDER_TEST_DATA = "./dataset/test_folder"
FOLDER_ISLANDVIEWER_DATA = "./dataset/prev_studies/islandviewer"
#FOLDER_TO_TRAIN_DATA_INFO = "./dataset/train_folder/info"
# define prefix for saving the training data
DATA_PREFIX = "testscript" #train, deduplicated, etc.

if __name__ == "__main__":

    os.makedirs(FOLDER_TRAIN_DATA, exist_ok=True)
    os.makedirs(FOLDER_TEST_DATA, exist_ok=True)
    os.makedirs(FOLDER_ISLANDVIEWER_DATA, exist_ok=True)

    islandviewer_file = f"{FOLDER_ISLANDVIEWER_DATA}/all_gis_islandviewer_iv4.csv"

    if not os.path.exists(islandviewer_file):
        print("====================Downloading Genomic Islands from IslandViewer4 database====================")
        # Downloading IslandViewer4 database
        # Step 1: Download
        url = "https://www.pathogenomics.sfu.ca/islandviewer/download/datasets/all_gis_islandviewer_iv4.csv.tar.gz"
        local_filename = wget.download(url, out=FOLDER_ISLANDVIEWER_DATA)

        # Step 2: Extract
        with tarfile.open(local_filename, "r:gz") as tar:
            tar.extractall(path=FOLDER_ISLANDVIEWER_DATA)
            print("\nExtraction complete.")

        extracted_filename = local_filename.removesuffix('.tar.gz')
        # Step 3: Load
        islandviewer_gis = pd.read_csv(extracted_filename)
    else:
        print("====================Reading Genomic Islands from IslandViewer4 database====================")
        islandviewer_gis = pd.read_csv(islandviewer_file)

    islandviewer_gis_stats = islandviewer_gis.groupby('accession').size().reset_index(name='n_samples').sort_values(by='n_samples', ascending=False)

    #print(islandviewer_gis_stats.head())

    # Download accession2taxid to get the taxonomy information for each accession
    if not os.path.exists(f"{FOLDER_ISLANDVIEWER_DATA}/nucl_gb.accession2taxid.gz"):
        print("====================Downloading accession2taxid info from NCBI====================")
        acc2taxid_url = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/accession2taxid/nucl_gb.accession2taxid.gz"
        acc2taxid_filename = wget.download(acc2taxid_url, out=FOLDER_ISLANDVIEWER_DATA)
    else:
        print("====================Reading accession2taxid info from NCBI====================")
        acc2taxid_filename = f"{FOLDER_ISLANDVIEWER_DATA}/nucl_gb.accession2taxid.gz"

    # Read the islandviewer with the taxonomy information
    taxid_file = f'{FOLDER_ISLANDVIEWER_DATA}/islandviewer_gis_taxid.xlsx'

    if not os.path.exists(taxid_file):
        # Create an SQLite database named "accessionTaxa.sqlite" to query from the accession2taxid data as it is a huge file
        acc2taxid_db = f"{FOLDER_ISLANDVIEWER_DATA}/accessionTaxa.sqlite"

        if not os.path.exists(acc2taxid_db):
            # required file: "nucl_gb.accession2taxid.gz"
            # output: SQLite database named "accessionTaxa.sqlite"
            print("====================Creating SQLite for accession2taxid info from NCBI====================")
            conn = sqlite3.connect(acc2taxid_db)
            cur = conn.cursor()

            cur.execute("CREATE TABLE IF NOT EXISTS accession2taxid (accession VARCHAR, accession_version VARCHAR, taxid INTEGER, gi INTEGER)")
            conn.commit()

            with gzip.open(acc2taxid_filename, "rt") as f:
                for line in f:
                    parts = line.strip().split('\t')
                    #print(parts)
                    cur.execute("INSERT INTO accession2taxid VALUES (?, ?, ?, ?)", (parts[0], parts[1], parts[2], int(parts[3]) if parts[3].isdigit() else None))

        print("====================Querying from SQLite accession2taxid info from NCBI====================")
        # once the SQLite database is created, we can query the database 
        conn = sqlite3.connect(acc2taxid_db)
        cur = conn.cursor()

        acc_ids = islandviewer_gis.accession.unique()  
        print(len(acc_ids))
        placeholders = ','.join('?' for _ in acc_ids)  # generates ?,?,?,? for param substitution
        query = f"SELECT * FROM accession2taxid WHERE accession_version IN ({placeholders})"
        #query = f"SELECT TOP 3 * FROM accession2taxid"
        cur.execute(query, acc_ids)
        results = cur.fetchall()

        # Create DataFrame with column names
        tax_id_df = pd.DataFrame(results, columns=['accession', 'accession_version', 'taxid', 'gi'])
        print(tax_id_df.head())
        # check remaining accessions without taxid
        remaining_accessions = islandviewer_gis_stats[~islandviewer_gis_stats['accession'].isin(tax_id_df['accession_version'].unique())]

        # Get the taxid for the remaining accession ids
        # - not all accessions in IslandViewer4 database are in the database
        # - query NCBI data for the remaining accession ids

        remaining_taxonomy_info = {}
        for acc in tqdm(remaining_accessions.accession.unique()):
            try:
                taxid = accession_to_taxid(acc)
                if taxid is not None:
                    remaining_taxonomy_info[acc] = taxid
            except Exception as e:
                pass

        clean_tax_info = {}
        error_tax_info = {}
        for key, val in remaining_taxonomy_info.items():
            try:
                clean_tax_info[key] = int(val)
            except:
                error_tax_info[key] = val
                pass 

        remaining_taxid = pd.DataFrame.from_dict(clean_tax_info, orient='index', columns=['taxid']).reset_index().rename(columns={'index': 'accession_version'})
        remaining_taxid['accession'] = remaining_taxid['accession_version'].str.split('.').str[0]
        remaining_taxid['gi'] = None
        # Append the new DataFrame to the existing tax_id_df
        full_tax_id_df = pd.concat([tax_id_df, remaining_taxid], ignore_index=True)

        # map the taxonomy info to the corresponding taxid 
        full_tax_id_df = full_tax_id_df.parallel_apply(map_taxon_to_df, axis=1)

        full_tax_id_df.to_excel(taxid_file, index=False)
    else:
        print("====================Reading taxonomy info for species in IslandViewer4====================")

        #read taxid dataframe
        full_tax_id_df = pd.read_excel(f'{FOLDER_ISLANDVIEWER_DATA}/islandviewer_gis_taxid.xlsx')

    #assign number of samples for each accession
    n_samples_per_accession_dict = islandviewer_gis_stats.set_index("accession")["n_samples"].to_dict()
    full_tax_id_df['n_samples'] = full_tax_id_df["accession_version"].map(n_samples_per_accession_dict)

    print("====================Sampling species from IslandViewer4====================")

    # sample some species based on WHO and RKI pathogen list
    # select genus based on WHO and RKI list
    who_list = ['Acinetobacter', 'Enterobacterales', 'Salmonella', 'Shigella', 'Enterococcus', 'Pseudomonas', 
                'Neisseria', 'Staphylococcus', 'Streptococcus', 'Haemophilus', 
                'Escherichia', 'Klebsiella', 'Enterobacter', 'Salmonella', 'Shigella', 'Citrobacter', 'Yersinia']

    rki_list = ['Acinetobacter', 'Burkholderia', 'Campylobacter', 'Citrobacter',
                    'Enterobacter', 'Enterococcus', 'Escherichia', 'Haemophilus',
                    'Helicobacter', 'Klebsiella', 'Morganella', 'Mycobacterium',
                    'Neisseria', 'Salmonella', 'Proteus', 'Providencia',
                    'Pseudomonas', 'Serratia', 'Shigella', 'Staphylococcus',
                    'Stenotrophomonas']

    who_rki_list = set(list(who_list + rki_list))
    who_rki_list = list(who_rki_list)

    who_rki_pool = full_tax_id_df[full_tax_id_df['genus'].isin(who_rki_list)]

    # each genus has 4 species on average -> sample 5 species per genus from the database
    sampled_species = sample_genus_species(who_rki_pool, n_species_per_genus=5)

    # save the sampled species to excel
    sampled_species.to_excel(f'{FOLDER_ISLANDVIEWER_DATA}/sampled_species.xlsx', index=False)

    ## Create training datasets with different species richness
    # - total 23 genera, 111 species (5 species per genus except 1 genus)
    # - case 1: vary species, vary samples
    # - case 2a: vary species (1,2,3,4,5 species per genus), fixed samples to 1500 (proportional to the original ratio of the species)
    # - case 2b: vary species (1,2,3,4,5 species per genus), fixed samples to 1500 (balanced ratio between species)
    # - case 3: fixed species to 111 species, vary total samples (1000,2000,3000,4000,5000)
    # - 1500 was chosen because the dataset with the fewest samples has 1426 samples
    # - this process resulted in 5 subcases per training case

    print("====================Sampling from Training Pool to creating training case1,2a,2b,3====================")
    
    os.makedirs(f"{FOLDER_TRAIN_DATA}/info", exist_ok=True)

    # Case 1
    sampled_case1 = {}
    for n in [1,2,3,4,5]:
        sampled_case1[n] = sample_n_subtaxa_per_taxa(sampled_species, taxa='genus', n_subtaxa=n, random_state=0)
        print(f'n_species: {len(sampled_case1[n])} with {sampled_case1[n].n_samples.sum()} samples')

    with open(f"{FOLDER_TRAIN_DATA}/info/{DATA_PREFIX}_case1.pkl", "wb") as f:
        pickle.dump(sampled_case1, f)  

    target_total_samples = 1500

    # Case 2a
    sampled_case2a = {}

    for n in [1,2,3,4,5]:
        #sampled_ = sample_n_species_per_genus(sampled_species, n_species=n, random_state=0)
        sampled_ = sample_n_subtaxa_per_taxa(sampled_species, taxa='genus',n_subtaxa=n, random_state=0)
        sampled_case2a[n] = proportional_caps(sampled_species, sampled_.species.unique() , target_total_samples)
        print(f'n_species: {len(sampled_case2a[n])} with {sampled_case2a[n].cap.sum()} samples')

    with open(f"{FOLDER_TRAIN_DATA}/info/{DATA_PREFIX}_case2a.pkl", "wb") as f:
        pickle.dump(sampled_case2a, f)  

    # Case 2b
    sampled_case2b = {}

    for n in [1,2,3,4,5]:
        sampled_ = sample_n_subtaxa_per_taxa(sampled_species, taxa='genus',n_subtaxa=n, random_state=0)
        sampled_case2b[n] = uniform_caps(sampled_species, sampled_.species.unique() , target_total_samples)
        print(f'n_species: {len(sampled_case2b[n])} with {sampled_case2b[n].cap.sum()} samples')

    with open(f"{FOLDER_TRAIN_DATA}/info/{DATA_PREFIX}_case2b.pkl", "wb") as f:
        pickle.dump(sampled_case2b, f)  

    # Case 3
    sampled_case3 = {}

    for n_samples in [1000,2000,3000,4000,5000]:
        #sampled_ = sample_n_species_per_genus(sampled_species, n_species=5, random_state=0)
        sampled_ = sample_n_subtaxa_per_taxa(sampled_species, taxa='genus',n_subtaxa=5, random_state=0)
        sampled_case3[n_samples//1000] = proportional_caps(sampled_species, sampled_.species.unique() , n_samples)
        print(f'n_species: {len(sampled_case3[n_samples//1000])} with {sampled_case3[n_samples//1000].cap.sum()} samples')

    with open(f"{FOLDER_TRAIN_DATA}/info/{DATA_PREFIX}_case3.pkl", "wb") as f:
        pickle.dump(sampled_case3, f)  

    # download FASTA files for the selected species
    sampled_acc_ids = sampled_species['accession_version'].unique().tolist()
    genome_folder = f'{FOLDER_ISLANDVIEWER_DATA}/genomes/sampled_species'

    # Run through all batches
    for batch in tqdm(batch_accessions(sampled_acc_ids, batch_size=5)):
        fetch_and_save_individual_fastas(batch, destination_folder=genome_folder)

    # write the training data pool
    training_pool_file = f"{FOLDER_TRAIN_DATA}/training_pool/training_pool.fasta"

    os.makedirs(f"{FOLDER_TRAIN_DATA}/training_pool",exist_ok=True)

    if not os.path.exists(training_pool_file):
        print("====================Writing Training Pool to a FASTA file====================")

        # perform negative sampling on the training data pool 
        sampled_gis = islandviewer_gis[islandviewer_gis['accession'].isin(sampled_acc_ids)]
        print(len(set(islandviewer_gis.accession.unique())&set(sampled_acc_ids)))
        training_pool_df = negatives_sampling(sampled_gis, genome_path = genome_folder)

        # write df to fasta file
        param = {'write_file':True,'filename':training_pool_file} #whether write sequences to a fasta file specified in filename
        fasta_data = training_pool_df.progress_apply(lambda x: df_to_fasta(x,
                                                                    dna_only=True, #dna_only determines whether or not to process sequences with IUPAC codes
                                                                    query_db=False, #query sequence from database
                                                                    **param), axis=1).to_list()
    else:
        print(f"{training_pool_file} already exists!")

    # print("====================Reading Training Pool FASTA file====================")

    # training_pool_df = fasta_to_df(training_pool_file)

    # print("====================Writing Training Case1====================")
    # #Case 1: vary species and samples

    # with open(f"{FOLDER_TRAIN_DATA}/info/{DATA_PREFIX}_case1.pkl", "rb") as f:
    #     selected_species_dict = pickle.load(f)

    # n_species_per_genus = selected_species_dict.keys()

    # for n in n_species_per_genus:
    #     selected_species = selected_species_dict[n].accession_version.tolist()
    #     pool_data_df_filtered = training_pool_df[training_pool_df['Accession'].isin(selected_species)]

    #     # write df to fasta file
    #     param = {'write_file':True,'filename':f"{FOLDER_TRAIN_DATA}/data/{DATA_PREFIX}_case1_{n}.fasta"} #whether write sequences to a fasta file specified in filename
    #     fasta_data = pool_data_df_filtered.progress_apply(lambda x: df_to_fasta(x,
    #                                                                 dna_only=True, #dna_only determines whether or not to process sequences with IUPAC codes
    #                                                                 query_db=False, #query sequence from database
    #                                                                 **param), axis=1)
    

    # print("====================Writing Training Case2a====================")
    # #Case 2a: balanced samples between datasets with proportional ratio to the original samples

    # with open(f"{FOLDER_TRAIN_DATA}/info/{DATA_PREFIX}_case2a.pkl", "rb") as f:
    #     selected_species_dict = pickle.load(f)

    # n_species_per_genus = selected_species_dict.keys()

    # for n in n_species_per_genus:
    #     selected_species = selected_species_dict[n].accession_version.tolist()
    #     sampled = training_pool_df[training_pool_df['Accession'].isin(selected_species)]
    #     # for each accession, sample up to cap number of samples per label
    #     cap_dict = selected_species_dict[n][['accession_version','cap']].set_index('accession_version').to_dict()['cap']

    #     pool_data_df_filtered = pd.DataFrame()
    #     for accession in cap_dict.keys():
    #         n_samples = cap_dict[accession]
    #         #positive samples
    #         pos_samples = sampled[(sampled['Accession']==accession) & (sampled['Label']=='1')].apply(lambda x: x.sample(n=min(len(x),n_samples), random_state=0))
    #         #negative samples
    #         neg_samples = sampled[(sampled['Accession']==accession) & (sampled['Label']=='0')].apply(lambda x: x.sample(n=min(len(x),n_samples), random_state=0))
    #         pool_data_df_filtered = pd.concat([pool_data_df_filtered, pos_samples, neg_samples], axis=0)

    #     # write df to fasta file
    #     param = {'write_file':True,'filename':f"{FOLDER_TRAIN_DATA}/data/{DATA_PREFIX}_case2a_{n}.fasta"} #whether write sequences to a fasta file specified in filename
    #     fasta_data = pool_data_df_filtered.progress_apply(lambda x: df_to_fasta(x,
    #                                                                 dna_only=True, #dna_only determines whether or not to process sequences with IUPAC codes
    #                                                                 query_db=False, #query sequence from database
    #                                                                 **param), axis=1)
        
    # print("====================Writing Training Case2b====================")
    # #Case 2b: balanced samples between datasets with balanced ratio between species

    # with open(f"{FOLDER_TRAIN_DATA}/info/{DATA_PREFIX}_case2b.pkl", "rb") as f:
    #     selected_species_dict = pickle.load(f)

    # n_species_per_genus = selected_species_dict.keys()

    # for n in n_species_per_genus:
    #     selected_species = selected_species_dict[n].accession_version.tolist()
    #     sampled = training_pool_df[training_pool_df['Accession'].isin(selected_species)]
    #     # for each accession, sample up to cap number of samples per label
    #     cap_dict = selected_species_dict[n][['accession_version','cap']].set_index('accession_version').to_dict()['cap']

    #     pool_data_df_filtered = pd.DataFrame()
    #     for accession in cap_dict.keys():
    #         n_samples = cap_dict[accession]
    #         #positive samples
    #         pos_samples = sampled[(sampled['Accession']==accession) & (sampled['Label']=='1')].apply(lambda x: x.sample(n=min(len(x),n_samples), random_state=0))
    #         #negative samples
    #         neg_samples = sampled[(sampled['Accession']==accession) & (sampled['Label']=='0')].apply(lambda x: x.sample(n=min(len(x),n_samples), random_state=0))
    #         pool_data_df_filtered = pd.concat([pool_data_df_filtered, pos_samples, neg_samples], axis=0)

    #     # write df to fasta file
    #     param = {'write_file':True,'filename':f"{FOLDER_TRAIN_DATA}/data/{DATA_PREFIX}_case2b_{n}.fasta"} #whether write sequences to a fasta file specified in filename
    #     fasta_data = pool_data_df_filtered.progress_apply(lambda x: df_to_fasta(x,
    #                                                                 dna_only=True, #dna_only determines whether or not to process sequences with IUPAC codes
    #                                                                 query_db=False, #query sequence from database
    #                                                                 **param), axis=1).to_list()
        

    # print("====================Writing Training Case3====================")
    # #Case 3: fixed species, vary samples

    # with open(f"{FOLDER_TRAIN_DATA}/info/{DATA_PREFIX}_case3.pkl", "rb") as f:
    #     selected_species_dict = pickle.load(f)

    # n_species_per_genus = selected_species_dict.keys()

    # for n in n_species_per_genus:
    #     selected_species = selected_species_dict[n].accession_version.tolist()
    #     sampled = training_pool_df[training_pool_df['Accession'].isin(selected_species)]
    #     # for each accession, sample up to cap number of samples per label
    #     cap_dict = selected_species_dict[n][['accession_version','cap']].set_index('accession_version').to_dict()['cap']

    #     pool_data_df_filtered = pd.DataFrame()
    #     for accession in cap_dict.keys():
    #         n_samples = cap_dict[accession]
    #         #positive samples
    #         pos_samples = sampled[(sampled['Accession']==accession) & (sampled['Label']=='1')].apply(lambda x: x.sample(n=min(len(x),n_samples), random_state=0))
    #         #negative samples
    #         neg_samples = sampled[(sampled['Accession']==accession) & (sampled['Label']=='0')].apply(lambda x: x.sample(n=min(len(x),n_samples), random_state=0))
    #         pool_data_df_filtered = pd.concat([pool_data_df_filtered, pos_samples, neg_samples], axis=0)

    #     # write df to fasta file
    #     param = {'write_file':True,'filename':f"{FOLDER_TRAIN_DATA}/data/{DATA_PREFIX}_case3_{n}.fasta"} #whether write sequences to a fasta file specified in filename
    #     fasta_data = pool_data_df_filtered.progress_apply(lambda x: df_to_fasta(x,
    #                                                                 dna_only=True, #dna_only determines whether or not to process sequences with IUPAC codes
    #                                                                 query_db=False, #query sequence from database
    #                                                                 **param), axis=1).to_list()
    

    # print("====================Create pre-defined folds for the training====================")
    # # create pre-defined folds for the training
    # train_cases = ['case1', 'case2a', 'case2b', 'case3']
    # n_subcases = 5

    # for train_case in train_cases:
    #     for n in range(1,n_subcases+1):
    #         print(f"{train_case}_{n}")

    #         folds_folder = f"{FOLDER_TRAIN_DATA}/folds"

    #         os.makedirs(folds_folder, exist_ok=True)

    #         folds_file = f"{folds_folder}/{DATA_PREFIX}_{train_case}_{n}_folds.json"

    #         if not os.path.exists(folds_file):
    #             # read data
    #             filename = f"{FOLDER_TRAIN_DATA}/data/{DATA_PREFIX}_{train_case}_{n}.fasta" 
    #             X, y, groups, genera, species = read_dataset(filename)

    #             n_splits = 5
    #             n_repeats = 5
    #             all_runs = create_fold(X, y, species, n_splits=n_splits, n_repeats=n_repeats)
                
                
    #             # Save to file
    #             with open(folds_file, "w") as f:
    #                 json.dump(all_runs, f, indent=2)
    #         else:
    #             print(f"{folds_file} already exists!")


