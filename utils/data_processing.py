# import libraries
import os, re
import pandas as pd 
import random
from Bio import Entrez
from Bio import SeqIO
import taxoniq

#====================general function for reading, writing sequences====================
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


def sample_n_subtaxa_per_taxa(meta_df, taxa='genus', n_subtaxa=1, min_pos_per_subtaxa=1, random_state=42):
    """
    Sample n subtaxa per taxa, optionally requiring a minimum number of positives.
    Example: sample n subspecies per species
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

def accession_to_taxid(acc):
    """Get TaxID for a given accession."""

    Entrez.email = "A.N.Other@example.com"  # Always tell NCBI who you are

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

def map_taxon_to_df(row):
    try:
        taxon = taxoniq.Taxon(row['taxid'])
        row['scientific_name'] = taxon.scientific_name

        for t in taxon.ranked_lineage:
            row[t.rank.name] = t.scientific_name 
    except Exception as e:
        row['scientific_name'] = None
    return row

def sample_genus_species(df, n_species_per_genus=5):

    # select one strain per subspecies with the most samples
    one_strain_per_subspecies = df.groupby('scientific_name').apply(pd.DataFrame.nlargest, n=1, columns=['n_samples']).reset_index(drop=True)

    #select unique species per genus
    sampled_genus = one_strain_per_subspecies.groupby('species').apply(pd.DataFrame.nlargest, n=1, columns=['n_samples']).reset_index(drop=True)

    #sample n species per genus
    sampled_species = sampled_genus.groupby('genus').apply(pd.DataFrame.nlargest, n=n_species_per_genus, columns=['n_samples']).reset_index(drop=True)

    return sampled_species


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

    Entrez.email = "A.N.Other@example.com"  # Always tell NCBI who you are

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