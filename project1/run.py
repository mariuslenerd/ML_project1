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

def train(x_train_sample, y_train_sample, initial_w, max_iters):

    w_least_squares, loss_ls = implemented_functions.least_squares(y_train_sample, x_train_sample)
    print(f"Least Squares: Loss = {loss_ls}")

    best_mse_coef = 0
    best_mse_loss = np.inf
    best_mse_weights = None
    for gamma in [0.01, 0.1, 0.5, 1]:
        w_sgd, loss_sgd = implemented_functions.mean_squared_error_sgd(y_train_sample,  x_train_sample, initial_w, max_iters, gamma)
        if loss_sgd[-1] < best_mse_loss:
            best_mse_loss = loss_sgd[-1]
            best_mse_coef = gamma
            best_mse_weights = w_sgd
        print(f"Mean Squared Error SGD (gamma={gamma}): Loss = {loss_sgd}")

    best_log_lasso_coef = 0
    best_log_lasso_loss = np.inf
    best_log_lasso_weights = None
    for gamma in [0.01, 0.1, 0.5]:
        for lambda_ in [0.1, 0.01]:
            w_log_lasso, loss_log_lasso = implemented_functions.reg_logistic_lasso_subgradient(y_train_sample, x_train_sample, lambda_, initial_w, max_iters, gamma)
            if loss_log_lasso < best_log_lasso_loss:
                best_log_lasso_loss = loss_log_lasso
                best_log_lasso_weights = w_log_lasso
            print(f"Regularized Lasso Logistic Regression (gamma={gamma}, lambda={lambda_}): Loss = {loss_log_lasso}")
            
    best_ridge_coef = 0
    best_ridge_loss = np.inf
    best_ridge_weights = None
    for ridge_coef in [0.001, 0.01, 0.1, 1, 10]:
        w_ridge, loss_ridge = implemented_functions.ridge_regression(y_train_sample, x_train_sample, ridge_coef)
        if loss_ridge < best_ridge_loss:
            best_ridge_loss = loss_ridge
            best_ridge_coef = ridge_coef
            best_ridge_weights = w_ridge
        print(f"Ridge Regression (alpha={ridge_coef}): Loss = {loss_ridge}")

    best_log_gamma = 0
    best_log_loss = np.inf
    best_log_weights = None
    for gamma in [0.01, 0.1, 0.5, 1]:
        w_logistic, loss_logistic = implemented_functions.logistic_regression(y_train_sample, x_train_sample, initial_w, max_iters, gamma)
        if loss_logistic < best_log_loss:
            best_log_loss = loss_logistic
            best_log_gamma = gamma
            best_log_weights = w_logistic
        print(f"Logistic Regression (gamma={gamma}): Loss = {loss_logistic}")

    best_log_reg_coef = 0
    best_log_reg_loss = np.inf
    best_log_reg_weights = None
    for gamma in [0.01, 0.1]:
        for reg_coef in [0.001, 0.01, 0.1]:
            w_reg_logistic, loss_reg_logistic = implemented_functions.reg_logistic_regression(y_train_sample, x_train_sample, reg_coef, initial_w, max_iters, gamma)
            if loss_reg_logistic < best_log_reg_loss:
                best_log_reg_loss = loss_reg_logistic
                best_log_reg_coef = reg_coef
                best_log_reg_weights = w_reg_logistic
            print(f"Regularized Logistic Regression (alpha={reg_coef}, gamma={gamma}): Loss = {loss_reg_logistic}")

    return w_sgd, w_least_squares, best_ridge_weights, best_log_weights, best_log_reg_weights, best_log_lasso_weights

def test_methods(x_test, y_test, w_sgd, w_least_squares, best_ridge_weights, best_log_weights, best_log_reg_weights, threshold=0.5):
    accuracy, y_pred = implemented_functions.compute_accuracy(y_test, x_test, w_sgd, "SGD", threshold)
    print(f"F1 score SGD is {implemented_functions.compute_f1_score(y_test, y_pred)}")
    accuracy, y_pred = implemented_functions.compute_accuracy(y_test, x_test, w_least_squares, "Least Squares", threshold)
    print(f"F1 score Least Squares is {implemented_functions.compute_f1_score(y_test, y_pred)}")
    accuracy, y_pred = implemented_functions.compute_accuracy(y_test, x_test, best_ridge_weights, "Ridge", threshold)
    print(f"F1 score Ridge is {implemented_functions.compute_f1_score(y_test, y_pred)}")
    accuracy, y_pred = implemented_functions.compute_accuracy(y_test, x_test, best_log_weights, "Logistic", threshold)
    print(f"F1 score Logistic is {implemented_functions.compute_f1_score(y_test, y_pred)}")
    accuracy, y_pred = implemented_functions.compute_accuracy(y_test, x_test, best_log_reg_weights, "Regularized Logistic", threshold)
    print(f"F1 score Regularized Logistic is {implemented_functions.compute_f1_score(y_test, y_pred)}")
    accuracy, y_pred = implemented_functions.compute_accuracy(y_test, x_test, best_log_lasso_weights, "Regularized Lasso", threshold)
    print(f"F1 score Regularized Lasso is {implemented_functions.compute_f1_score(y_test, y_pred)}")
    return y_pred

def test(x_test, y_test, w_sgd, w_least_squares, best_ridge_weights, best_log_weights, best_log_reg_weights, best_log_lasso_weights, threshold=0.5):
    
    print("------Testing with threshold 0.5------")
    test_methods(x_test_sample, y_test_sample, w_sgd, w_least_squares, best_ridge_weights, best_log_weights, best_log_reg_weights, threshold=0.5)
    print("------Testing with threshold 0.3------")
    test_methods(x_test_sample, y_test_sample, w_sgd, w_least_squares, best_ridge_weights, best_log_weights, best_log_reg_weights, threshold=0.3)
    print("------Testing with threshold 0.7------")
    test_methods(x_test_sample, y_test_sample, w_sgd, w_least_squares, best_ridge_weights, best_log_weights, best_log_reg_weights, threshold=0.7)

    sgd_best_threshold = test_thresholds(x_test_sample, y_test_sample, w_sgd, "SGD")
    least_squares_best_threshold = test_thresholds(x_test_sample, y_test_sample, w_least_squares, "Least Squares")
    logistic_best_threshold = test_thresholds(x_test_sample, y_test_sample, best_log_weights, "Logistic")
    logistic_reg_best_threshold = test_thresholds(x_test_sample, y_test_sample, best_log_reg_weights, "Regularized Logistic")
    ridge_best_threshold = test_thresholds(x_test_sample, y_test_sample, best_ridge_weights, "Ridge")
    log_lasso_best_threshold = test_thresholds(x_test_sample, y_test_sample, best_log_lasso_weights, "Regularized Lasso")


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
#x_train_raw, x_test_raw,y_train_raw, train_ids, test_ids = load_csv_data('dataset')

print("Preprocessing data...")

#y_train_original = y_train_raw.copy()
#y_train_raw[y_train_raw == -1] = 0
#replace NaN by a float to treat NaN as a categorical feature
#TODO: NAN'S : -10 or 0 --> SAME BETWEEN TRAIN AND TEST 
#x_test_raw = np.nan_to_num(x_test_raw, nan = -10.0)

data_annoted = data_preprocessing.read_annotated_csv('dataset/data_anotated.csv')
#x_train, y_train, x_test = data_preprocessing.preprocess_data2(x_train_raw[:,:], y_train_raw[:], x_test_raw[:,:], data_annoted)

print("Saving preprocessed data...")
#np.savetxt('x_train_preprocessed.csv', x_train, delimiter=',')
#np.savetxt('y_train_preprocessed.csv', y_train, delimiter=',')
#np.savetxt('x_test_preprocessed.csv', x_test, delimiter=',')

x_train = np.loadtxt('x_train_preprocessed.csv', delimiter=',')
x_test = np.loadtxt('x_test_preprocessed.csv', delimiter=',')
y_train = np.loadtxt('y_train_preprocessed.csv', delimiter=',')

initial_w = np.zeros(x_train.shape[1])
max_iters = 50

x_test_sample = x_train[-40000:,:]
y_test_sample = y_train[-40000:]
x_train_sample = x_train[:-40000,:]
y_train_sample = y_train[:-40000]

print("Starting training...")
w_sgd, w_least_squares, best_ridge_weights, best_log_weights, best_log_reg_weights, best_log_lasso_weights = train(x_train_sample, y_train_sample, initial_w, max_iters)

print("Starting testing...")
test(x_test_sample, y_test_sample, w_sgd, w_least_squares, best_ridge_weights, best_log_weights, best_log_reg_weights, best_log_lasso_weights, threshold=0.5)
