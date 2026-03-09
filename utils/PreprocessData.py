from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import itertools
from . import FileProcessing,CheckAccPseParameter


class PreprocessData:

    def __init__(self,parameters):
        self.parameters = parameters

    def generate_kmers(self, segment):
        ''' Generate overlapping kmers from a given DNA sequence
           segment : DNA sequence
        '''
        kmers = []
        for i in range(0, len(segment) - (self.parameters.KMER_SIZE - 1)):
            start = i
            end = i + self.parameters.KMER_SIZE
            kmer = segment[start:end]
            kmers.append(kmer)

        return kmers

    def get_complete_sequence_kmers(self, dna_sequence):
        '''
        Generates kmers for the entire sequence
        :param dna_sequence: DNA sequence
        :return: kmers of the entire DNA sequence
        '''
        sequence = str(dna_sequence.seq).lower()
        kmers = [self.generate_kmers(sequence)]
        return kmers
    
    def replace_iupac_with_nucleotide(self, sequence):
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

    def split_dna_sequence(self, dna_sequence):
        '''
        Divides the DNA segment into equal small segments of sizes self.window_size
        '''
        
        sequence = str(dna_sequence.seq).upper()
        # Check if the sequence contains any IUPAC codes
        # if len(set(sequence) - set({'A','T','G','C'})) != 0:
        #     sequence = self.replace_iupac_with_nucleotide(sequence)

        segments = []
        segment_borders = []
        dna_sequence_tqdm = tqdm(range(0, len(sequence), self.parameters.WINDOW_SIZE), position=0, leave=True)
        for i in dna_sequence_tqdm:
            start = i
            if (i + self.parameters.WINDOW_SIZE) < len(sequence):
                end = i + self.parameters.WINDOW_SIZE
            else:
                end = len(sequence)
            segment = sequence[start:end]
            segments.append(segment)

            segment_borders.append([start, end])

        return segments, segment_borders
    
    def encode_sequence(self, input):
        '''
        Encode the input sequences according to the representation of choice
        '''
        
        desc = self.parameters.REPRESENTATION
        desc_default_para = self.parameters.DESC_DEFAULT_PARA

        if isinstance(input,str) and input.endswith('fasta'):
            descriptor = FileProcessing.Descriptor(input, desc_default_para)
        else:
            descriptor = FileProcessing.Descriptor("", desc_default_para)

            if isinstance(input,str):
                input = [input]

            seq = [["Name",seq,"Label","testing"] for seq in input]
            descriptor.assign_fasta(seq)

        
        if desc in ['DAC', 'TAC']:
            my_property_name, my_property_value, my_kmer, ok = CheckAccPseParameter.check_acc_arguments(desc, descriptor.sequence_type, desc_default_para)
            status = descriptor.make_ac_vector(my_property_name, my_property_value, my_kmer)
        elif desc in ['DCC', 'TCC']:
            my_property_name, my_property_value, my_kmer, ok = CheckAccPseParameter.check_acc_arguments(desc, descriptor.sequence_type, desc_default_para)
            status = descriptor.make_cc_vector(my_property_name, my_property_value, my_kmer)
        elif desc in ['DACC', 'TACC']:
            my_property_name, my_property_value, my_kmer, ok = CheckAccPseParameter.check_acc_arguments(desc, descriptor.sequence_type, desc_default_para)
            status = descriptor.make_acc_vector(my_property_name, my_property_value, my_kmer)
        elif desc in ['PseDNC', 'PseKNC', 'PCPseDNC', 'PCPseTNC', 'SCPseDNC', 'SCPseTNC']:
            my_property_name, my_property_value, ok = CheckAccPseParameter.check_Pse_arguments(desc, descriptor.sequence_type, desc_default_para)
            cmd = 'descriptor.' + desc + '(my_property_name, my_property_value)'
            status = eval(cmd)
        else:
            cmd = 'descriptor.' + desc + '()'
            status = eval(cmd)

        X = descriptor.encoding_array[1:][:,2:].astype(float)
        #y = descriptor.encoding_array[1:][:,1].astype(int)
        #groups = np.array(['_'.join(label.split('_')[:2]) for label in descriptor.encoding_array[1:][:,0]])

        return X
    
