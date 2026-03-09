#import libraries
import argparse
import os 
import numpy as np 
import pandas as pd
from utils.data_processing import read_results, read_dataset, read_fold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import time 
import os, sys
pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, default="benbow") 
    parser.add_argument("--data-folder", type=str, default="dataset/train_folder")
    parser.add_argument("--output-folder", type=str, default="outputs/crossval") 
    parser.add_argument("--predictions-folder", type=str, default="outputs/predictions/deduplicated_samples")
    args = parser.parse_args()
    return args

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
    

if __name__ == "__main__":
# required files: 
# FASTA file for training data
# pre-defined folds in JSON format
# predictions in Excel format

    start_time = time.time()
    print('--- Start ---')

    # read data for each fold
    args = get_args()
    filename = args.filename
    data_folder = args.data_folder 
    predictions_folder = args.predictions_folder
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)

    # check the prediction files in the prediction_dir

    pred_files = set([f.split('_predictions')[0] for f in os.listdir(predictions_folder)])
    pred_files = sorted(pred_files)

    #for filename in pred_files:
    print(f"1. Read Data {filename}")
    # Read training data
    train_filename = f"{data_folder}/data/{filename}.fasta"
    X_train, y_train, groups_train, genera_train, species_train = read_dataset(train_filename)

    print("2. Read Pre-defined Folds")
    # Read fold
    fold_filename = f"{data_folder}/folds/{filename}_folds.json"
    labels = read_fold(fold_filename)

    # read predictions
    print("3. Read predictions of the cross-validation")
    
    files = [f for f in os.listdir(predictions_folder) if os.path.isfile(os.path.join(predictions_folder, f))]

    predictions_df = pd.DataFrame()

    for file in files:
        if file.endswith('.xlsx') and file.startswith(f'{filename}_predictions'):
            filename = os.path.join(predictions_folder,file)
            df = read_results(filename, header=['fold','representation','model'])
            predictions_df = pd.concat([predictions_df,df])

    ## Filter out models that are not of interest (e.g., ensemble methods)
    #predictions_df = predictions_df[~predictions_df['model'].isin(['AdaBoost','GradientBoosting','XGBoost','Bagging'])]

    # Create a label column to identify each (feature_a, feature_b) pair
    predictions_df['pair_id'] = predictions_df['representation'].astype(str) + '/' + predictions_df['model'].astype(str)

    print(f"4. Evaluate cross-validation results for {filename} dataset")
    eval_metrics = compute_metrics(predictions_df, labels, y_train)


    output_path = os.path.join(output_folder,f'{filename}_crossval.xlsx')
    print(f"5. Saving cross-validation results to {output_path}")
    eval_metrics.to_excel(output_path, index=False)

    finish_time = time.time()
    print('--- Finish ---')
    print(' --- Process took {:.3f} seconds ---'.format(finish_time-start_time))






