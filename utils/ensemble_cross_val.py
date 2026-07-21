import numpy as np
import random
from sklearn.pipeline import Pipeline # type: ignore
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler
from utils.Parameters import Parameters
from sklearn.model_selection import StratifiedGroupKFold, BaseCrossValidator
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier,RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_score, f1_score, fbeta_score, recall_score
from sklearn.utils import check_random_state
from utils.util import specificity_score, true_positive, true_negative, false_positive, false_negative
from scipy.stats import mode 
from .util import get_representations
from tqdm import tqdm
from functools import partial

import os, sys, re
from pathlib import Path
file_path = os.path.split(os.path.realpath(__file__))[0]
pPath = Path(file_path).parent
sys.path.append(pPath)

import logging 
logger = logging.getLogger('cross_val')

class RepeatedStratifiedGroupKFold(BaseCrossValidator):
    def __init__(self, n_splits=5, n_repeats=10, random_state=None, bins=10):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.bins = bins

    def split(self, X, y, groups):
        #rng = check_random_state(self.random_state)
        #unique_groups = np.unique(groups)

        for repeat in range(self.n_repeats):
            # Shuffle group order for randomness
            #shuffled_groups = rng.permutation(unique_groups)

            # Estimate label ratio (e.g. proportion of positive samples per group)
            #group_ratios = []
            #for group in shuffled_groups:
            #    idx = np.where(groups == group)[0]
            #    ratio = np.sum(y[idx]) / len(idx)
            #    group_ratios.append(ratio)
            #group_ratios = np.array(group_ratios)

            # Bin ratios for stratification
            #stratify_labels = np.digitize(group_ratios, np.linspace(0, 1, self.bins))

            #skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=rng)
            random_state = random.getrandbits(16)
            skf = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=random_state)
            for train_idx, test_idx in skf.split(X, y, groups):
                yield train_idx, test_idx

            #for train_groups_idx, test_groups_idx in skf.split(shuffled_groups, stratify_labels):
            #    train_groups = shuffled_groups[train_groups_idx]
            #    test_groups = shuffled_groups[test_groups_idx]

            #    train_idx = np.where(np.isin(groups, train_groups))[0]
            #    test_idx = np.where(np.isin(groups, test_groups))[0]

            #    yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits * self.n_repeats

def transform_sequences(X, representation, desc_default_para):
    try:
        X,_,_ = get_representations(X, representation, desc_default_para)

        return X
    except Exception as e:
        raise ValueError("Error in get_representations: {}".format(e))

def get_custom_transformer(representation):
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
            desc_default_para.update({key:para_dict[representation][key]})

    # Wrap it in a transformer that binds the extra arguments
    # custom_transformer = FunctionTransformer(
    #     func=lambda X: transform_sequences(X, representation=representation, desc_default_para=desc_default_para),
    #     validate=False
    # )
    custom_transformer = FunctionTransformer(
        func = partial(transform_sequences, representation=representation, desc_default_para=desc_default_para),
        validate = False
    )

    return custom_transformer

def get_pipeline(model, representation, scaler_type=None, default=True):
    """
    Create a pipeline for the specified model with the given transformer.
    :param model: The model type (e.g., 'SVC', 'RandomForest').
    :param representation: The representation type (e.g., 'Z_curve', 'Kmer-6').
    :param scaler_type: The scaler type (e.g., 'minmax' or 'zscore')
    :return: A Pipeline instance for the specified model.
    """
    custom_transformer = get_custom_transformer(representation)

    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'zscore':
        scaler = StandardScaler()
    else:
        scaler = None

    if model == 'SVC' or model == 'SVM':
        if default:
            clf = SVC(probability=True, random_state=42)
        else:
            clf = SVC(probability=True, kernel='rbf', C=2, gamma='scale', random_state=42)
    elif model == 'RandomForest':
        # clf = RandomForestClassifier(n_estimators=500, oob_score=True,
        #                              criterion='gini', max_depth=10, max_features='sqrt',
        #                              random_state=42)
        # clf = RandomForestClassifier(criterion= 'entropy',
        #                             max_depth= None,
        #                             max_features= 'sqrt',
        #                             min_samples_split= 5,
        #                             n_estimators= 500, random_state=42)
        clf = RandomForestClassifier(random_state=42)
    elif model == 'NaiveBayes':
        clf = GaussianNB()
    elif model == 'LogisticRegression':
        clf = LogisticRegression(random_state=42)
    elif model == 'DecisionTree':
        clf = DecisionTreeClassifier(random_state=42)
    elif model == 'AdaBoost':
        clf = AdaBoostClassifier(algorithm='SAMME')
    elif model == 'GradientBoosting':
        clf = GradientBoostingClassifier()
    elif model == 'Bagging':
        clf = BaggingClassifier()
    elif model == 'XGBoost':
        clf = XGBClassifier()
    else:
        raise ValueError("Unsupported model type: {}".format(model))

    if not scaler:
        pipeline = Pipeline([
                ('transformer', custom_transformer),
                ('scaler', scaler),
                (model, clf)
            ])
    else:
        pipeline = Pipeline([
                ('transformer', custom_transformer),
                (model, clf)
            ]) 

    return pipeline


def run_cross_validation(train_params):
    """
    a function to generate predictions from pre-defined cross-validation folds

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

def get_classifiers(pairs, scaler_type=None, default=True):
    classifiers = []

    if len(pairs) == 1:
        representation, model = pairs[0].split('/')
        pipeline = get_pipeline(model, representation, scaler_type, default)
        return pipeline

    for pair in pairs:
        representation, model = pair.split('/')
        pipeline = get_pipeline(model, representation, scaler_type, default)
        classifiers.append((pair,pipeline))
    return classifiers

def get_single_model(candidates, scaler_type=None, default=True):
    """
    returns a single model given the candidate model

    Parameters: 
    ----------
    candidates : list, a list of candidates for the single model e.g. ['RCKmer-7/SVM']

    Returns: 
    -------
    model : classifier, a single classifier
    """
    logger.info(f"get single model with candidates: {candidates}")

    # get list of candidate models
    model = get_classifiers(candidates, scaler_type)

    if len(candidates) != 1:
        raise ValueError("Expected exactly one candidate for single model, got {}".format(len(candidates)))
    
    return model

def get_ensemble_model(candidates, ensemble_type='stacking', default=True):
    """
    returns an ensemble model given the candidate models and ensemble type

    Parameters: 
    ----------
    candidates : list, a list of candidates for the ensemble model e.g. ['RCKmer-7/SVM', 'Subsequence/RandomForest']

    Returns: 
    -------
    ensemble : classifier, an ensemble classifier
    """

    # get list of candidate models
    models = get_classifiers(candidates, default=default)

    logger.info(f"get ensemble model with candidates: {candidates} and ensemble type: {ensemble_type}")
    meta_models = {
            "stacking": StackingClassifier(estimators=None, final_estimator=LogisticRegression()),
            "voting_soft": VotingClassifier(estimators=None, voting="soft"),
            "voting_hard": VotingClassifier(estimators=None, voting="hard")
        }
    ensemble = meta_models[ensemble_type]
    ensemble.estimators = models
    return ensemble

def compute_eval_score(y_true, y_pred):
    """
    returns evaluation scores given true and predicted labels

    Parameters: 
    ----------
    y_true : array, true labels
    y_pred : array, predicted labels 

    Returns: 
    -------
    eval_scores: dict, a dictionary of evaluation scores
    """

    logger.info("compute evaluation scores")
    eval_scores = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
            'F_1': f1_score(y_true, y_pred), 
            'F_beta_0.5': fbeta_score(y_true, y_pred, beta=0.5),
            'F_beta_2': fbeta_score(y_true, y_pred, beta=2),
            'MCC': matthews_corrcoef(y_true, y_pred),
            'TP': true_positive(y_true, y_pred),
            'TN': true_negative(y_true, y_pred),
            'FP': false_positive(y_true, y_pred),
            'FN': false_negative(y_true, y_pred),
    }
    return eval_scores
