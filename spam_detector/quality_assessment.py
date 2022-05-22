import os
import pandas as pd
from datetime import datetime
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix


def run_experiment(config, model_name, model, return_predictions=False):
    print("Testing model {}\n".format(model_name))
    print("Loading test dataset")
    test = pd.read_csv(os.path.join(config['dataset_build']['res_dataset_folder_name'], 'test.csv'))
    print("Test dataset size = {} entries".format(len(test)))
    print("Share of spam = {0:.4f}\n".format(test['label'].mean()))

    # calculate predictions and measure runtime
    print("Generating predictions")
    start_time = datetime.now()
    y_pred = model.predict(test[['message_id', 'subject', 'text']])
    end_time = datetime.now()
    print("Predictions generated\n")
    runtime = end_time - start_time
    runtime_per_1000 = 1000 * runtime / len(test)
    print('Total inference time: {}'.format(runtime))
    print('Inference time per 1000 entries: {}\n'.format(runtime_per_1000))

    # binary classification metrics
    y_true = test['label'].values
    print('Binary classification metrics:')
    # roc-auc as a general quality metric
    roc_auc = roc_auc_score(y_true, y_pred)
    print("roc-auc score: {0:.4f}".format(roc_auc))
    # calculate precision at a chosen recall level
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    i = 0
    for i in range(len(recall) - 1):
        if recall[i] <= config['testing']['recall_level']: break
    selected_threshold = thresholds[i]
    selected_recall = recall[i]
    selected_precision = precision[i]
    print('At recall {0:.4f} precision = {1:.4f}\n'.format(selected_recall, selected_precision))
    print('Confusion matrix:\n{}'.format(confusion_matrix(y_true, y_pred >= selected_threshold)))

    results = {'model_name': model_name, 'test_dataset': {'size': len(test), 'spam_share': test['label'].mean()},
            'runtime': {'total_runtime': runtime, 'runtime_per_1000': runtime_per_1000},
            'quality_metrics': {'roc_auc_score': roc_auc, 'threshold': selected_threshold,
                                'precision': selected_precision, 'recall': selected_recall}}
    if return_predictions:
        results['predictions'] = y_pred
        results['labels'] = y_true

    return results