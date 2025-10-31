"""Some helper functions for project 1."""

import csv
import numpy as np
import os
from implemented_functions import *
import plots

def load_csv_data(data_path, sub_sample=False):
    """
    This function loads the data and returns the respectinve numpy arrays.
    Remember to put the 3 files in the same folder and to not change the names of the files.

    Args:
        data_path (str): datafolder path
        sub_sample (bool, optional): If True the data will be subsempled. Default to False.

    Returns:
        x_train (np.array): training data
        x_test (np.array): test data
        y_train (np.array): labels for training data in format (-1,1)
        train_ids (np.array): ids of training data
        test_ids (np.array): ids of test data
    """
    y_train = np.genfromtxt(
        os.path.join(data_path, "y_train.csv"),
        delimiter=",",
        skip_header=1,
        dtype=int,
        usecols=1,
    )
    x_train = np.genfromtxt(
        os.path.join(data_path, "x_train.csv"), delimiter=",", skip_header=1
    )
    x_test = np.genfromtxt(
        os.path.join(data_path, "x_test.csv"), delimiter=",", skip_header=1
    )

    train_ids = x_train[:, 0].astype(dtype=int)
    test_ids = x_test[:, 0].astype(dtype=int)
    x_train = x_train[:, 1:]
    x_test = x_test[:, 1:]

    # sub-sample
    if sub_sample:
        y_train = y_train[::50]
        x_train = x_train[::50]
        train_ids = train_ids[::50]

    return x_train, x_test, y_train, train_ids, test_ids


def create_csv_submission(ids, y_pred, name):
    """
    This function creates a csv file named 'name' in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})

def best_by_method_to_array(best_by_method):
    """
    Convert best_by_method dict into a structured NumPy array.
    Columns: method, degree, lambda, gamma, test_loss
    """
    rows = []
    for method_name, info in best_by_method.items():
        rows.append([
            method_name,
            info["degree"],
            info["lambda"],
            info["gamma"],
            info["test_loss"],
        ])
    return np.array(rows, dtype=object)



def test_thresholds(x_test, y_test, weights, method):
    f1_scores = []
    accuracies = []
    best_threshold = 0
    best_weighted_f1 = 0
    for threshold in np.linspace(0, 1, 20):
        accuracy, y_pred = compute_accuracy(y_test, x_test, weights, method, threshold, detailed=False)
        f1_score = compute_f1_score(y_test, y_pred)
        f1_scores.append(f1_score)
        if (5*f1_score + accuracy) >= best_weighted_f1:
            best_weighted_f1 = 5*f1_score + accuracy
            best_threshold = threshold
        accuracies.append(accuracy)
    plots.plot_threshold(best_threshold, f1_scores, accuracies, method)
    print("The best threshold is:", best_threshold)
    return best_threshold

def load_best_params(file_path):
    data = np.loadtxt(file_path, delimiter=',', dtype=str)
    headers = data[0]
    rows = data[1:]
    
    best_params = [dict(zip(headers, row)) for row in rows]
    return best_params

def compute_f1_score(y_true, y_pred):
    """
    Compute the F1 score given true and predicted binary labels.

    Args:
        y_true: numpy array of shape (N,), true binary labels (0 or 1)
        y_pred: numpy array of shape (N,), predicted binary labels (0 or 1)
    Returns:
        f1: float, the F1 score
    """
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))
    f1 = 2*true_positives / (2*true_positives + false_positives + false_negatives)
    return f1

def compute_accuracy(y_test, x_test, w, method, threshold=0.5, mode=None, detailed=True):
    """
    Compute accuracy of predictions on test data.
    Args:
        y_test: numpy array of shape (N,), true binary labels (0 or 1)
        x_test: numpy array of shape (N,D), input data
        w: numpy array of shape (D,), model parameters
        method: str, method name ("logistic", "Regularized Logistic", etc.)
        threshold: float, decision threshold for classification
        mode: str or None, if 'submission', transform 0 predictions to -1
        detailed: bool, if True, print accuracy
    Returns:
        computed_accuracy: float, accuracy of predictions
        y_pred: numpy array of shape (N,), predicted binary labels (0 or 1)
    """
    
    if method in ["logistic", "Regularized Logistic", "reg_lasso_logistic"]:
        y_pred = sigmoid(x_test@w)
    else:
        y_pred = x_test@w
    y_pred[y_pred <= threshold] = 0
    y_pred[y_pred > threshold] = 1
    if mode == 'submission':
        # transform all 0 predictions to -1 for submission format
        y_pred[y_pred == 0] = -1
        return 0, y_pred
    computed_accuracy = np.sum(y_pred == y_test)/len(y_test)
    if detailed:
        print(f"Accuracy of {method} is {computed_accuracy*100}%")
    return computed_accuracy, y_pred
