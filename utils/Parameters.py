#reference: https://github.com/Superzchen/iLearnPlus
class Parameters:
    def __init__(self):
        '''Intitialize parameters'''
        self.WINDOW_SIZE = 10000
        self.KMER_SIZE = 6
        self.UPPER_THRESHOLD = 0.80
        self.LOWER_THRESHOLD = 0.50
        self.TUNE_METRIC = 1000
        self.MINIMUM_GI_SIZE = 10000
        self.REPRESENTATION = 'RCKmer'
        self.DESC_DEFAULT_PARA = {             
            # default parameter for descriptors
            'sliding_window': 5,
            'kspace': 3,
            'props': ['CIDH920105', 'BHAR880101', 'CHAM820101', 'CHAM820102', 'CHOC760101', 'BIGC670101', 'CHAM810101', 'DAYM780201'],
            'nlag': 2,
            'weight': 0.05,
            'lambdaValue': 3,
            'PseKRAAC_model': 'g-gap',
            'g-gap': 2,
            'k-tuple': 2,
            'RAAC_clust': 1,
            'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101',
            'kmer': 7,
            'mismatch': 1,
            'delta': 1,
            'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise',
            'Tri-DNA-Phychem': 'Dnase I;Bendability (DNAse)',
            'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)',
            'distance': 0,
            'cp': 'cp(20)',
            'k_max': 7,
            'k_start':1,
            'k_default': 3
        }  

        self.PARA_DICT = {
            'EAAC': {'sliding_window': 5},
            'CKSAAP': {'kspace': 3},
            'EGAAC': {'sliding_window': 5},
            'CKSAAGP': {'kspace': 3},
            'AAIndex': {'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101'},
            'NMBroto': {'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101', 'nlag': 3, 'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise',},
            'Moran': {'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101', 'nlag': 3, 'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise',},
            'Geary': {'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101', 'nlag': 3, 'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise',},
            'KSCTriad': {'kspace': 3},
            'SOCNumber': {'nlag': 3},
            'QSOrder': {'nlag': 3, 'weight': 0.05},
            'PAAC': {'weight': 0.05, 'lambdaValue': 3},
            'APAAC': {'weight': 0.05, 'lambdaValue': 3},
            'DistancePair': {'distance': 0, 'cp': 'cp(20)',},
            'AC': {'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101', 'nlag': 3},
            'CC': {'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101', 'nlag': 3},
            'ACC': {'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101', 'nlag': 3},
            'PseKRAAC type 1': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 2': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 3A': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 3B': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 4': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 5': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 6A': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 6B': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 6C': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 7': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 8': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 9': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 10': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 11': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 12': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 13': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 14': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 15': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 16': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'Kmer': {'kmer': 3},
            'RCKmer': {'kmer': 3},
            'Mismatch': {'kmer': 3, 'mismatch': 1},
            'Subsequence': {'kmer': 3, 'delta': 0},
            'ENAC': {'sliding_window': 5},
            'CKSNAP': {'kspace': 3},
            'DPCP': {'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise', 'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)'},
            'DPCP type2': {'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise', 'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)'},
            'TPCP': {'Tri-DNA-Phychem': 'Dnase I;Bendability (DNAse)'},
            'TPCP type2': {'Tri-DNA-Phychem': 'Dnase I;Bendability (DNAse)'},
            'DAC': {'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise', 'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)', 'nlag': 3},
            'DCC': {'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise', 'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)', 'nlag': 3},
            'DACC': {'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise', 'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)', 'nlag': 3},
            'TAC': {'Tri-DNA-Phychem': 'Dnase I;Bendability (DNAse)', 'nlag': 3},
            'TCC': {'Tri-DNA-Phychem': 'Dnase I;Bendability (DNAse)', 'nlag': 3},
            'TACC': {'Tri-DNA-Phychem': 'Dnase I;Bendability (DNAse)', 'nlag': 3},
            'PseDNC': {'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise', 'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)', 'weight': 0.05, 'lambdaValue': 3},
            'PseKNC': {'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise', 'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)', 'weight': 0.05, 'lambdaValue': 3, 'kmer': 3},
            'PCPseDNC': {'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise', 'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)', 'weight': 0.05, 'lambdaValue': 3},
            'PCPseTNC': {'Tri-DNA-Phychem': 'Dnase I;Bendability (DNAse)', 'weight': 0.05, 'lambdaValue': 3},
            'SCPseDNC': {'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise', 'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)', 'weight': 0.05, 'lambdaValue': 3},
            'SCPseTNC': {'Tri-DNA-Phychem': 'Dnase I;Bendability (DNAse)', 'weight': 0.05, 'lambdaValue': 3},
        }

    def set_window_size(self, window_size):
        self.WINDOW_SIZE = window_size

    def set_kmer_size(self, kmer_size):
        self.KMER_SIZE = kmer_size

    def set_upper_threshold(self, upper_threshold):
        self.UPPER_THRESHOLD = upper_threshold

    def set_lower_threshold(self, lower_threshold):
        self.LOWER_THRESHOLD = lower_threshold

    def set_tune_metric(self, tune_metric):
        self.TUNE_METRIC = tune_metric

    def set_minimum_gi_size(self, minimum_gi_size):
        self.MINIMUM_GI_SIZE = minimum_gi_size
    
    def set_representation(self, representation):
        self.REPRESENTATION = representation

    def set_desc_default_para(self, params:dict):
        self.DESC_DEFAULT_PARA.update(params)

