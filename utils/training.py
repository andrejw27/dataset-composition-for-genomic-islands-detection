import random 
import numpy as np
import json
from Bio import SeqIO
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import (make_scorer, matthews_corrcoef,confusion_matrix, fbeta_score,
                            accuracy_score, precision_score, recall_score, f1_score)
from . import FileProcessing, CheckAccPseParameter
import os, sys
from pathlib import Path
file_path = os.path.split(os.path.realpath(__file__))[0]
pPath = Path(file_path).parent
sys.path.append(pPath)

import logging 
logger = logging.getLogger('training')

# ====================create pre-defined fold for each dataset====================
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

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
            # Split the data into train and test sets
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            groups_train, groups_test = groups[train_idx], groups[test_idx]
            
            print(f"Outer Fold {fold_idx}: |overlap groups| = {len(set(groups_train) & set(groups_test))}")

            # split X_train to train and validation sets with groups stratified and take only 1 split
            cv_inner = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=rng)
            inner_valid_idx, inner_test_idx = next(cv_inner.split(X_test, y_test, groups_test))

            valid_idx = test_idx[inner_valid_idx]
            test_idx = test_idx[inner_test_idx]

            print(f"Inner Fold {fold_idx}: |overlap groups| = {len(set(groups[valid_idx]) & set(groups[test_idx]))}")

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

    # return unique species (not subspecies)
    unique_species = [' '.join(sp.split(' ')[:2]) for sp in species_test]
    X_test = np.array(X_test).astype(str)
    y_test = np.array(y_test).astype(int)
    groups_test = np.array(groups_test)
    genera_test = np.array(genera_test)
    species_test = np.array(unique_species)
    return X_test, y_test, groups_test, genera_test, species_test

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
    'Accuracy': make_scorer(accuracy_score),
    'Precision': make_scorer(precision_score, zero_division=0),
    'Recall': make_scorer(recall_score, zero_division=0),
    'Specificity': make_scorer(specificity_score, zero_division=0),
    'f1': 'f1',  # F-beta with beta=1 is equivalent to F1-score
    #'f_0.5': make_scorer(fbeta_score, beta=0.5),
    #'f_2': make_scorer(fbeta_score, beta=2),
    'MCC': make_scorer(matthews_corrcoef),
    'TP': make_scorer(true_positive),
    'TN': make_scorer(true_negative),
    'FP': make_scorer(false_positive),
    'FN': make_scorer(false_negative),
}


def predict_output_new(train_params):
    """
    a function to evaluate model given train  data set and test data set

    Parameters: 
    ----------
    train_params : dict, parameters for training include data representation, model

    Returns: 
    -------
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
            'NaiveBayes': GaussianNB(),
            # 'AdaBoost': AdaBoostClassifier(random_state=42),
            # 'GradientBoosting': GradientBoostingClassifier(),
            # 'Bagging':BaggingClassifier(),
            # 'XGBoost': XGBClassifier()
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


def hyperparameter_tuning(train_params):
    
    # Hyperparameter grid 
    param_grid_search = {
        'RandomForest':{
            'n_estimators': [100, 200, 500, 1000],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10, 20],
            'criterion': ['entropy','gini'],
            'max_features': ['sqrt', 'log2'],
        },
        'SVM':{
            'C': [0.1, 1, 10],
            'kernel': ['rbf'],
            'gamma': ['scale', 0.01, 1.0]
        },
        'LogisticRegression':{
            'penalty':['l1','l2','elasticnet','none'],
            'C' : np.logspace(-4,4,20),
            'solver': ['lbfgs','newton-cg','liblinear','sag','saga'],
            'max_iter'  : [100,1000,2500,5000]
        }
    }

    # # for script testing only
    # param_grid_search = {
    #     'RandomForest':{
    #         'n_estimators': [100, 200],
    #         'max_depth': [None],
    #         'min_samples_split': [2],
    #         'min_samples_leaf': [2],
    #         'criterion': ['gini'],
    #         'max_features': ['sqrt'],
    #     },
    #     'SVM':{
    #         'C': [10],
    #         'kernel': ['rbf'],
    #         'gamma': ['scale']
    #     },
    #     'LogisticRegression':{
    #         'penalty':['l1'],
    #         'C' : np.logspace(-4,4,20),
    #         'solver': ['lbfgs'],
    #         'max_iter'  : [100]
    #     }
    # }

    # Define the classifier
    clf = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42),
    }

    train_file = train_params['train_file']
    test_data = train_params['test_data']
    model = train_params['model']
    
    if 'k' in train_params.keys():
        train_params['representation_params'].update({'kmer':train_params['k']})

    if train_params['representation'] in ['Kmer', 'RCKmer']:
        key = "{}-{}".format(train_params['representation'],train_params['representation_params']['kmer'])
    else:
        key = train_params['representation']

    logger.info('representation Train Data with {}'.format(key))
    X_train,y_train,groups_train = get_representations(train_file, 
                                                       train_params['representation'], 
                                                       train_params['representation_params'])

    train_dataname = train_file.split('/')[-1]
    train_dataname = train_dataname.split('.')[0]

    logger.info('Grid Search {} {} {}'.format(train_dataname,model,key))

    # Define StratifiedGroupKFold
    cv = StratifiedGroupKFold(n_splits=5)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=clf[model], 
                               param_grid=param_grid_search[model], 
                               cv=cv, 
                               n_jobs=4,
                               scoring=scoring, 
                               refit='f1')

	# execute search
    result = grid_search.fit(X_train, y_train, groups=groups_train)

	# get the best performing model fit on the whole training set
    best_model = result.best_estimator_

    final_results = {}

    for test_file in test_data.keys():
        logger.info('representation Test Data {} with {}'.format(test_file,key))
        X_test,y_test,groups_test = test_data[test_file]
        
        # evaluate model on the hold out dataset
        y_pred = best_model.predict(X_test)

        eval_scores = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred), 
            'f_0.5': fbeta_score(y_test, y_pred, beta=0.5),
            'f_2': fbeta_score(y_test, y_pred, beta=2),
            'MCC': matthews_corrcoef(y_test, y_pred),
            'TP': true_positive(y_test, y_pred),
            'TN': true_negative(y_test, y_pred),
            'FP': false_positive(y_test, y_pred),
            'FN': false_negative(y_test, y_pred),
        }

        test_dataname = test_file.split('/')[-1]
        test_dataname = test_dataname.split('.')[0]

        final_results[f"{train_dataname}/{test_dataname}"] = eval_scores

    return {f'{train_dataname}_{model}_{key}':{'eval_scores': final_results,
                                          'cv_results':grid_search.cv_results_,
                                          'best_params':grid_search.best_params_,
                                          'best_index':grid_search.best_index_}}

def task_hyperparameter_tuning(hpo_params):
    file = hpo_params['filename']
    model = hpo_params['model']
    representation = hpo_params['representation']

    if 'kwargs' in hpo_params.keys():
        kwargs = hpo_params['kwargs']
    else:
        kwargs = {}

    grid_search_res = hyperparameter_tuning(file, model, representation, **kwargs)

    return grid_search_res