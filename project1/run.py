import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import *
import importlib
importlib.reload(data_preprocessing)
import cross_validation
importlib.reload(cross_validation)
from implemented_functions import *
from helpers import *
import PCA
importlib.reload(PCA)
import os

"""
print("Loading data...")
x_train_raw, x_test_raw,y_train_raw, train_ids, test_ids = load_csv_data('project1/dataset')
print("Preprocessing data...")
data_annoted = data_preprocessing.read_annotated_csv('project1/dataset/data_anotated.csv')
x_train, y_train, x_test_final = data_preprocessing.preprocess_data(x_train_raw[:,:], y_train_raw[:], x_test_raw[:,:], data_annoted)

print("Saving preprocessed data to CSV...")
np.savetxt('project1/dataset/preprocessed/x_train_preprocessed.csv', x_train, delimiter=',')
np.savetxt('project1/dataset/preprocessed/y_train_preprocessed.csv', y_train, delimiter=',')
np.savetxt('project1/dataset/preprocessed/x_test_preprocessed.csv', x_test_final, delimiter=',')
"""

print("Loading data...")
x_train = np.loadtxt('project1/dataset/preprocessed/x_train_preprocessed.csv', delimiter=',')
y_train = np.loadtxt('project1/dataset/preprocessed/y_train_preprocessed.csv', delimiter=',')
x_test_final = np.loadtxt('project1/dataset/preprocessed/x_test_preprocessed.csv', delimiter=',')

y_train[y_train == -1] = 0

print("Loading best parameters...")
best_params_dict = load_best_params('project1/results_cross_val_plain.csv')

print("Data loaded.")
#x_train, x_test_final, _, _, _, _, _ = PCA.PCA_threshold(x_train, x_test_final, 0.95)
#print(x_train.shape)

x_train,x_test,y_train,y_test = data_preprocessing.split_data(x_train, y_train, 0.8, seed=1)
max_iters = 50

functions = {
    "least_squares": class_weighted_least_squares,
    "mse_sgd": mean_squared_error_sgd,
    "reg_lasso_logistic": reg_logistic_lasso,
    "ridge": ridge_regression,
    "logistic": logistic_regression,
    "reg_logistic": reg_logistic_regression,
}

models = ["least_squares", "mse_sgd", "reg_lasso_logistic", "ridge", "logistic"]

results = {}  

for model in models:
    model_param = next(d for d in best_params_dict if d['method'] == model)
    degree = int(model_param['degree'])
    
    x_tr = data_preprocessing.build_poly(x_train, degree)
    x_te = data_preprocessing.build_poly(x_test, degree)
    initial_w = np.zeros(x_tr.shape[1])

    print(f"Model: {model}, Degree: {degree}, x_tr shape: {x_tr.shape}, x_te shape: {x_te.shape}")

    args = {"y": y_train, "tx": x_tr, "initial_w": initial_w}
    if "lambda" in model_param and model_param["lambda"] not in [None, "None", np.str_("None")]:
        args["lambda_"] = float(model_param["lambda"])
    if "gamma" in model_param and model_param["gamma"] not in [None, "None", np.str_("None")]:
        args["gamma"] = float(model_param["gamma"])
    if "max_iters" in locals():
        args["max_iters"] = max_iters

    w, loss = functions[model](**{k: v for k, v in args.items() if k in functions[model].__code__.co_varnames})

    # store everything neatly
    results[model] = {
        "weight": w,
        "loss": loss,
        "x_te": x_te,
        "degree": degree,
    }

# === Threshold selection and F1 evaluation ===
best_thresholds = {}
for i, (model, info) in enumerate(results.items(), start=1):
    print(f"Threshold {i}: {model}")
    best_thresholds[model] = test_thresholds(info["x_te"], y_test, info["weight"], model)



for model, info in results.items():
    threshold = best_thresholds[model]
    accuracy, y_pred = compute_accuracy(y_test, info["x_te"], info["weight"], model, threshold)
    f1 = compute_f1_score(y_test, y_pred)
    print(f"F1 score {model}: {f1:.4f}")


# This part of the code can be used to generate predictions for the ai crowd test set once the best model and threshold are selected
for model, info in results.items():
    print("Generating predictions for test set...")
    x_test_poly = data_preprocessing.build_poly(x_test_final, int(info['degree']))
    acc, y_pred_test = compute_accuracy(None, x_test_poly, info["weight"], model, best_thresholds[model], mode = 'submission')

    ids = [i for i in range(328135, 437513+1)]
    create_csv_submission(ids, y_pred_test, 'project1/results/submission_' + model + '.csv')
