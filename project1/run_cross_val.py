import numpy as np
from cross_validation import cross_validation_demo_all,plot_cv_results
import pandas as pd

def best_by_method_to_dataframe(best_by_method):
    """
    Turn best_by_method dict into a pandas DataFrame with columns:
    method, degree, lambda, gamma, test_loss
    """
    rows = []
    for method_name, info in best_by_method.items():
        rows.append({
            "method": method_name,
            "degree": info["degree"],
            "lambda": info["lambda"],
            "gamma": info["gamma"],
            "test_loss": info["test_loss"],
        })
    df = pd.DataFrame(rows)
    return df

def curves_to_dataframe(curves):
    """
    Flatten the full `curves` dict into a long-form DataFrame.

    Output columns:
    method, degree, lambda, gamma, train_loss, test_loss

    Notes:
    - For methods that only use lambda: gamma is NaN
    - For methods that only use gamma:  lambda is NaN
    - For methods with neither: both lambda and gamma are NaN
    - For methods with both: we output one row per (gamma, lambda)
    """
    rows = []

    for method_name, per_degree in curves.items():
        for degree, data in per_degree.items():

            has_lambda = "lambdas" in data
            has_gamma  = "gammas" in data

            if has_lambda and has_gamma:
                # 2D grid case
                lambdas_arr = data["lambdas"]
                gammas_arr  = data["gammas"]
                train_grid  = data["train"]  # shape [len(gamma), len(lambda)]
                test_grid   = data["test"]

                for gi, gamma_val in enumerate(gammas_arr):
                    for li, lambda_val in enumerate(lambdas_arr):
                        rows.append({
                            "method": method_name,
                            "degree": degree,
                            "lambda": float(lambda_val),
                            "gamma":  float(gamma_val),
                            "train_loss": float(train_grid[gi, li]),
                            "test_loss":  float(test_grid[gi, li]),
                        })

            elif has_lambda and not has_gamma:
                # 1D lambda sweep
                lambdas_arr = data["lambdas"]
                train_arr   = data["train"]
                test_arr    = data["test"]

                for li, lambda_val in enumerate(lambdas_arr):
                    rows.append({
                        "method": method_name,
                        "degree": degree,
                        "lambda": float(lambda_val),
                        "gamma":  np.nan,
                        "train_loss": float(train_arr[li]),
                        "test_loss":  float(test_arr[li]),
                    })

            elif has_gamma and not has_lambda:
                # 1D gamma sweep
                gammas_arr = data["gammas"]
                train_arr  = data["train"]
                test_arr   = data["test"]

                for gi, gamma_val in enumerate(gammas_arr):
                    rows.append({
                        "method": method_name,
                        "degree": degree,
                        "lambda": np.nan,
                        "gamma":  float(gamma_val),
                        "train_loss": float(train_arr[gi]),
                        "test_loss":  float(test_arr[gi]),
                    })

            else:
                # no hyperparams case
                rows.append({
                    "method": method_name,
                    "degree": degree,
                    "lambda": np.nan,
                    "gamma":  np.nan,
                    "train_loss": float(data["train"]),
                    "test_loss":  float(data["test"]),
                })

    return pd.DataFrame(rows)


print("Loading data...")
x_train = np.loadtxt('project1/x_train_preprocessed.csv', delimiter=',')
y_train = np.loadtxt('project1/y_train_preprocessed.csv', delimiter=',')

##only on a subset##
x_train_sub = x_train[:10000,:]
y_train_sub = y_train[:10000]
degrees   = [2, 3, 4, 5]
k_fold    = 5
lambdas   = [10e-4,10e-3,10e-2,10e-1,1]
gammas    = [10e-4,10e-3,10e-2,10e-1,1]
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

df = best_by_method_to_dataframe(best_by_method)
df.to_csv('results_cross_val', index=False)

df_curves = curves_to_dataframe(curves)
df_curves.to_csv('curves_cross_val', index=False)

plot_cv_results(curves, method="mse_gd", degree=5)
plot_cv_results(curves, method="mse_sgd", degree=5)
plot_cv_results(curves, method="least_squares", degree=5)
plot_cv_results(curves, method="reg_logistic", degree=5)
plot_cv_results(curves, method="ridge", degree=5)
plot_cv_results(curves, method="reg_lasso_logistic", degree=5)

