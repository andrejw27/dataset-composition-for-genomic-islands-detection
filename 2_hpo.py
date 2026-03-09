import time
import pickle
import re
from tqdm import tqdm
import argparse
from utils.Parameters import Parameters

import os, sys
pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)

from utils.training import hyperparameter_tuning, get_representations

from logging_config import setup_logging
import logging

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-worker", type=int, default=1)
    #parser.add_argument("--filename", type=str, default="benbow")
    parser.add_argument("--train-folder", type=str, default="dataset/train_folder")
    parser.add_argument("--test-folder", type=str, default="dataset/test_folder")
    args = parser.parse_args()
    return args

args = get_args()
filename = args.filename

logger = logging.getLogger(__name__)

log_path = os.path.join(pPath,'logs/hpo')

if not os.path.exists(log_path):
    logger.info('Creating Logging Folder')
    os.mkdir(log_path)

log_filename = os.path.join(log_path,'{}.log'.format(filename))
setup_logging(log_filename=log_filename)

def main():

    #filename = args.filename
    #parameters
    train_folder = args.train_folder
    test_folder = args.test_folder
    train_files = [f"{train_folder}/data/{file}" for file in os.listdir(f"{train_folder}/data")]
    train_files = sorted(train_files)
    test_files = [f"{test_folder}/{file}" for file in os.listdir(f"{test_folder}")]
    representation = "RCKmer"
    k = 7
    model = "SVM"

    #parameters for features encoding
    parameters = Parameters()
    desc_default_para = parameters.DESC_DEFAULT_PARA
    para_dict = parameters.PARA_DICT

    # copy parameters for each descriptor
    if representation in para_dict:
        for key in para_dict[representation]:
            desc_default_para[key] = para_dict[representation][key]
    
    if representation in ['Kmer', 'RCKmer']:
        #k = int(representation.split('-')[-1])
        train_params = {
                'representation': representation,
                'representation_params': desc_default_para,
                'k': k
        }

    else:
        train_params = {
            'representation': representation,
            'representation_params': desc_default_para,
        }

    test_data = {}

    if 'k' in train_params.keys():
        train_params['representation_params'].update({'kmer':train_params['k']})
        
    if train_params['representation'] in ['Kmer', 'RCKmer']:
        key = "{}-{}".format(train_params['representation'],train_params['representation_params']['kmer'])
    else:
        key = train_params['representation']

    for test_file in test_files:
        logger.info(f"Transforming {test_file} with {key}")
        X_test,y_test,groups_test = get_representations(test_file, 
                                                        train_params['representation'], 
                                                        train_params['representation_params'])
        
        test_data[test_file] = (X_test, y_test, groups_test)

    for train_file in tqdm(train_files):
        logger.info(f"Training on {train_file}")

        params = {
            'train_file': train_file,
            'test_data': test_data,
            'model': model,
        }
        params.update(train_params)

        grid_search_results = hyperparameter_tuning(params)
        

        logger.info('Done Hyperparameter Tuning')

        hpo_path = os.path.join(pPath,'hpo')

        if not os.path.exists(hpo_path):
            logger.info('Creating HPO Folder')
            os.mkdir(hpo_path)

        logger.info('Saving Output')

        for key in grid_search_results:
            with open(os.path.join(hpo_path,f"{key}.pkl"), 'wb') as f:
                pickle.dump(grid_search_results, file=f)

        logger.info('Done')

if __name__=="__main__":
    start_time = time.time()
    logger.info('--- Start ---')
    main()
    finish_time = time.time()
    logger.info('--- Finish ---')
    logger.info(' --- Process took {:.3f} seconds ---'.format(finish_time-start_time))