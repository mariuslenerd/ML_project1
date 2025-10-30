import numpy as np
from cross_validation import cross_validation_demo_all
from plots import plot_cv_results
from helpers import best_by_method_to_array



print("Loading data...")
x_train = np.loadtxt('project1/x_train_preprocessed.csv', delimiter=',')
y_train = np.loadtxt('project1/y_train_preprocessed.csv', delimiter=',')
y_train[y_train == -1] = 0

# Use subset
x_train_sub = x_train[:50000, :]
y_train_sub = y_train[:50000]
degrees   = [1,2,3,4]
k_fold    = 5
lambdas   = [10e-4,10e-3,10e-2,10e-1,1,10e1,10e2]
gammas    = [10e-3,10e-2,10e-1,1]
max_iters = 100
initial_w = np.zeros(x_train_sub.shape[1])

best_by_method, curves = cross_validation_demo_all(
    y=y_train_sub,
    x=x_train_sub,
    degrees=degrees,
    k_fold=k_fold,
    lambdas=lambdas,
    initial_w=initial_w,
    max_iters=max_iters,
    gammas=gammas,
    seed=42,
    verbose=True,
)

# Convert to arrays
best_array = best_by_method_to_array(best_by_method)

np.savetxt(
    "results_cross_val_plain.csv",
    best_array,
    fmt="%s",
    delimiter=",",
    header="method,degree,lambda,gamma,test_loss",
    comments=""
)

for method_name, info in best_by_method.items():
    best_degree = info["degree"]
    plot_cv_results(curves, method=method_name, degree=best_degree)

