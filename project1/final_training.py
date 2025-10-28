import numpy as np
import matplotlib.pyplot as plt
from helpers import *
import data_preprocessing
import importlib
importlib.reload(data_preprocessing)
import cross_validation
#import TEST
#importlib.reload(TEST)
importlib.reload(cross_validation)
import implemented_functions
importlib.reload(implemented_functions)
import PCA
importlib.reload(PCA)
import pandas as pd 


def test_thresholds(x_test, y_test, weights, method):
    f1_scores = []
    accuracies = []
    best_threshold = 0
    for threshold in np.linspace(0, 1, 20):
        accuracy, y_pred = implemented_functions.compute_accuracy(y_test, x_test, weights, method, threshold, detailed=False)
        f1_score = implemented_functions.compute_f1_score(y_test, y_pred)
        f1_scores.append(f1_score)
        if f1_score >= max(f1_scores):
            best_threshold = threshold
        accuracies.append(accuracy)
        #print(f"-----{method}: F1 score {method} at threshold {threshold} is {implemented_functions.compute_f1_score(y_test, y_pred)}")
    plt.axvline(x=best_threshold, color='r', linestyle='--')
    plt.plot(np.linspace(0, 1, 20), f1_scores, marker='o')
    plt.title(f'F1 Score vs Threshold for {method}')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.plot(np.linspace(0, 1, 20), accuracies, marker='o', color='orange')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.show()
    print("The best threshold is:", best_threshold)
    return best_threshold

print("Loading data...")
x_train = np.loadtxt('project1/x_train_preprocessed.csv', delimiter=',')
y_train = np.loadtxt('project1/y_train_preprocessed.csv', delimiter=',')

x_train, _, _, _, _, _ = PCA.PCA_threshold(x_train, 0.95)
print(x_train.shape)

print("training1 ")
x_train,x_test,y_train,y_test = data_preprocessing.split_data(x_train, y_train, 0.8, seed=1)
max_iters = 100 
x_tr_ls = data_preprocessing.build_poly(x_train,2)
x_te_ls = data_preprocessing.build_poly(x_test,2)
initial_w = np.zeros(x_tr_ls.shape[1])
print("training2 ")
w_weighted_least_squares, loss_weighted_ls = implemented_functions.class_weighted_least_squares(y_train, x_tr_ls)
x_tr_sgd = data_preprocessing.build_poly(x_train,3)
x_te_sgd = data_preprocessing.build_poly(x_test,3)
initial_w = np.zeros(x_tr_sgd.shape[1])
print("training3 ")
w_sgd, loss_sgd = implemented_functions.mean_squared_error_sgd(y_train,  x_tr_sgd, initial_w, max_iters, 0.001)
x_tr_lasso = data_preprocessing.build_poly(x_train,2)
x_te_lasso = data_preprocessing.build_poly(x_test,2)
initial_w = np.zeros(x_tr_lasso.shape[1])
print("training4 ")
w_log_lasso, loss_log_lasso = implemented_functions.reg_logistic_lasso_subgradient(y_train, x_tr_lasso, 0.001, initial_w, max_iters, 0.001)
x_tr_ridg= data_preprocessing.build_poly(x_train,4)
x_te_ridg = data_preprocessing.build_poly(x_test,4)
initial_w = np.zeros(x_tr_ridg.shape[1])
print("training5 ")
w_ridge, loss_ridge = implemented_functions.ridge_regression(y_train, x_tr_ridg, 0.01)
x_tr_log= data_preprocessing.build_poly(x_train,4)
x_te_log = data_preprocessing.build_poly(x_test,4)
initial_w = np.zeros(x_tr_log.shape[1])
print("training6 ")
w_logistic, loss_logistic = implemented_functions.logistic_regression(y_train, x_tr_log, initial_w, max_iters, 0.1)
x_tr_reg_log= data_preprocessing.build_poly(x_train,4)
x_te_reg_log = data_preprocessing.build_poly(x_test,4)
initial_w = np.zeros(x_tr_reg_log.shape[1])
#print("training7 ")
#w_reg_logistic, loss_reg_logistic = implemented_functions.reg_logistic_regression(y_train, x_tr_reg_log, 0.001, initial_w, max_iters, 0.1)

print("threshold1 ")
sgd_best_threshold = test_thresholds(x_te_sgd, y_test, w_sgd, "SGD")
print("threshold2 ")
least_squares_best_threshold = test_thresholds(x_te_ls, y_test, w_weighted_least_squares, "Least Squares")
print("threshold3 ")
logistic_best_threshold = test_thresholds(x_te_log, y_test, w_logistic, "Logistic")
print("threshold4 ")
#logistic_reg_best_threshold = test_thresholds(x_te_reg_log, y_test, w_reg_logistic, "Regularized Logistic")
print("threshold5 ")
ridge_best_threshold = test_thresholds(x_te_ridg, y_test, w_ridge, "Ridge")
print("threshold6 ")
log_lasso_best_threshold = test_thresholds(x_te_lasso, y_test, w_log_lasso, "Regularized Lasso")

accuracy, y_pred = implemented_functions.compute_accuracy(y_test, x_te_sgd, w_sgd, "SGD", sgd_best_threshold)
print(f"F1 score SGD is {implemented_functions.compute_f1_score(y_test, y_pred)}")
accuracy, y_pred = implemented_functions.compute_accuracy(y_test, x_te_ls, w_weighted_least_squares, "Least Squares", least_squares_best_threshold)
print(f"F1 score Least Squares is {implemented_functions.compute_f1_score(y_test, y_pred)}")
accuracy, y_pred = implemented_functions.compute_accuracy(y_test, x_te_ridg, w_ridge, "Ridge", ridge_best_threshold)
print(f"F1 score Ridge is {implemented_functions.compute_f1_score(y_test, y_pred)}")
accuracy, y_pred = implemented_functions.compute_accuracy(y_test, x_te_log, w_logistic, "Logistic", logistic_best_threshold)
print(f"F1 score Logistic is {implemented_functions.compute_f1_score(y_test, y_pred)}")
#accuracy, y_pred = implemented_functions.compute_accuracy(y_test, x_te_reg_log, w_reg_logistic, "Regularized Logistic", logistic_reg_best_threshold)
print(f"F1 score Regularized Logistic is {implemented_functions.compute_f1_score(y_test, y_pred)}")
accuracy, y_pred = implemented_functions.compute_accuracy(y_test, x_te_lasso, w_log_lasso, "Regularized Lasso", log_lasso_best_threshold)
print(f"F1 score Regularized Lasso is {implemented_functions.compute_f1_score(y_test, y_pred)}")