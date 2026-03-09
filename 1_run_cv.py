
import itertools
import json
import time
from tqdm import tqdm
import argparse
from multiprocessing import Pool
import multiprocessing
import os, sys
from utils.data_processing import multiindex_dict_to_df
from utils.training import predict_output_new
from utils.Parameters import Parameters
from logging_config import setup_logging
import logging

pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation-index", type=int, default=1)
    parser.add_argument("--n-worker", type=int, default=1)
    parser.add_argument("--filename", type=str, default="deduplicated_case1_1")
    parser.add_argument("--train-folder", type=str, default="dataset/train_folder")
    args = parser.parse_args()
    return args

args = get_args()
filename = args.filename
idx = args.representation_index

logger = logging.getLogger(__name__)

log_path = os.path.join(pPath,'logs/predictions/{}'.format(filename))

if not os.path.exists(log_path):
    logger.info('Creating Logging Folder')
    os.makedirs(log_path)

log_filename = os.path.join(log_path,'{}_{}.log'.format(filename,idx))
setup_logging(log_filename=log_filename)

def main():

    train_folder = args.train_folder
    train_files = [f"{train_folder}/data/{filename}.fasta"]
    
    with open(f"{train_folder}/folds/{filename}_folds.json","r") as f:
        folds = json.load(f)

    representations_dict = {
        1: ['NAC','CKSNAP','Subsequence','ASDC','Mismatch', 'GC'],
        2: ['Z_curve_9bit', 'Z_curve_12bit','Z_curve_36bit','Z_curve_48bit','Z_curve_144bit'],
        3: ['PseEIIP','DAC','DCC','DACC','TAC','TCC','TACC','Moran','Geary','NMBroto','DPCP','TPCP'],
        4: ['MMI','PseDNC','PseKNC','PCPseDNC','PCPseTNC','SCPseDNC','SCPseTNC'],
        5: ['Kmer'],
        6: ['RCKmer'],
    }

    idx = args.representation_index

    logger.info('Working Path: {}'.format(pPath))

    representations = representations_dict[idx]

    logger.info('Representations: {}'.format(representations))
          
    list_train_params = []

    for elements in itertools.product(train_files, representations) :
        (train_file, representation) = elements

        #parameters for features encoding
        parameters = Parameters()
        desc_default_para = parameters.DESC_DEFAULT_PARA
        para_dict = parameters.PARA_DICT

        # copy parameters for each descriptor
        if representation in para_dict:
            for key in para_dict[representation]:
                desc_default_para[key] = para_dict[representation][key]
        
        if representation in ['Kmer', 'RCKmer']:
            for k in range(1,8):
                train_params = {
                        'train_file': train_file,
                        'folds': folds,
                        'representation': representation,
                        'representation_params': desc_default_para,
                        'k': k
                }

                list_train_params.append(train_params)
        else:
            train_params = {
                'train_file': train_file,
                'folds': folds,
                'representation': representation,
                'representation_params': desc_default_para,
            }

            list_train_params.append(train_params)

    if len(list_train_params)>1 and args.n_worker > 1:
        n_worker = multiprocessing.cpu_count()
        n_worker = min(args.n_worker,n_worker)

        with Pool(n_worker) as p:
            results = p.map(predict_output_new, tqdm(list_train_params))
    else:
        results = [predict_output_new(list_train_params[0])]

    logger.info('Done Predicting')

    predictions = {}

    for res in results:
        predictions.update(res)

    predictions_df = multiindex_dict_to_df(predictions)
    
    predictions_path = os.path.join(pPath,'predictions')

    if not os.path.exists(predictions_path):
        logger.info('Creating Predictions Folder')
        os.makedirs(predictions_path)

    logger.info('Saving Output')
    prediction_file = os.path.join(predictions_path,f"{filename}_predictions_{idx}.xlsx")
    predictions_df.to_excel(prediction_file)

    logger.info('Predictions Path: {}'.format(prediction_file))

    logger.info('Done')

if __name__=="__main__":
    start_time = time.time()
    logger.info('--- Start ---')
    main()
    finish_time = time.time()
    logger.info('--- Finish ---')
    logger.info(' --- Process took {:.3f} seconds ---'.format(finish_time-start_time))