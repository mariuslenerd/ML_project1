import numpy as np
from implemented_functions import *
from data_preprocessing import build_poly
from plots import plot_all_methods

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, k_indices, k, lambda_, degree, initial_w, max_iters, gamma):

    te_idx = k_indices[k]
    tr_idx = np.hstack([k_indices[i] for i in range(k_indices.shape[0]) if i != k])

    x_tr_k = x[tr_idx]
    y_tr_k = y[tr_idx]
    x_te_k = x[te_idx]
    y_te_k = y[te_idx]

    x_tr_k_poly = build_poly(x_tr_k,degree)
    x_te_k_poly = build_poly(x_te_k,degree)

    # --- Train all models on the training fold ---
    w_mse_gd, _loss_mse_gd   = mean_squared_error_gd(y_tr_k,  x_tr_k_poly, initial_w, max_iters, gamma)
    w_mse_sgd, _loss_mse_sgd = mean_squared_error_sgd(y_tr_k, x_tr_k_poly, initial_w, max_iters, gamma)
    w_rdg, _loss_rdg         = ridge_regression(y_tr_k,       x_tr_k_poly, lambda_)
    w_ls, _loss_ls           = least_squares(y_tr_k,          x_tr_k_poly)
    w_lr, _loss_lr           = logistic_regression(y_tr_k,    x_tr_k_poly, initial_w, max_iters, gamma)
    w_reg_lr, _loss_reg_lr   = reg_logistic_regression(y_tr_k,x_tr_k_poly, lambda_, initial_w, max_iters, gamma)

    results = {}

    # Regression models -> RMSE
    results["mse_gd"]       = (compute_rmse(y_tr_k, x_tr_k_poly, w_mse_gd),   compute_rmse(y_te_k, x_te_k_poly, w_mse_gd))
    results["mse_sgd"]      = (compute_rmse(y_tr_k, x_tr_k_poly, w_mse_sgd),  compute_rmse(y_te_k, x_te_k_poly, w_mse_sgd))
    results["least_squares"]= (compute_rmse(y_tr_k, x_tr_k_poly, w_ls),       compute_rmse(y_te_k, x_te_k_poly, w_ls))
    results["ridge"]        = (compute_rmse(y_tr_k, x_tr_k_poly, w_rdg),      compute_rmse(y_te_k, x_te_k_poly, w_rdg))

    # Logistic models -> log-loss (y must be 0/1 for these)
    results["logistic"]     = (compute_logistic_loss(y_tr_k, x_tr_k_poly, w_lr),    compute_logistic_loss(y_te_k, x_te_k_poly, w_lr))
    results["reg_logistic"] = (compute_reg_logistic_loss(y_tr_k, x_tr_k_poly, w_reg_lr, lambda_), compute_reg_logistic_loss(y_te_k, x_te_k_poly, w_reg_lr, lambda_))

    return results



### CROSS VALIDATION DEMO (ALL MODELS) FOR ONE DEGREE AND ONE LEARNING RATE GAMMA ###
def cross_validation_demo(y, x, degree, k_fold, lambdas, initial_w, max_iters, gamma, seed=12):
    """
    Cross-validate *all* models you trained inside cross_validation()
    across the provided 'lambdas'. For models that don't depend on lambda,
    the curve will be (nearly) flat across lambdas.
    Returns:
        best_by_method: dict method -> (best_lambda, best_test_loss)
    """

    k_indices = build_k_indices(y, k_fold, seed)

    methods = ["mse_gd","mse_sgd","least_squares","ridge","logistic","reg_logistic"]
    curves_tr = {m: [] for m in methods}
    curves_te = {m: [] for m in methods}

    for lambda_ in lambdas:
        sums_tr = {m: 0.0 for m in methods}
        sums_te = {m: 0.0 for m in methods}

        for k in range(k_fold):
            print(f"lambda={lambda_}, fold={k}")
            print("'\n")
            res = cross_validation(y, x, k_indices, k, lambda_, degree, initial_w, max_iters, gamma)
            for m in methods:
                tr_k, te_k = res[m]
                sums_tr[m] += tr_k
                sums_te[m] += te_k

        for m in methods:
            curves_tr[m].append(sums_tr[m] / k_fold)
            curves_te[m].append(sums_te[m] / k_fold)

    # Choose “best” lambda per method by min test loss
    best_by_method = {}
    for m in methods:
        arr = np.array(curves_te[m])
        best_idx = int(np.argmin(arr))
        best_by_method[m] = (lambdas[best_idx], arr[best_idx])


    plot_all_methods(lambdas, curves_tr, curves_te)

    # summary
    print(f"Degree = {degree}, K = {k_fold}")
    for m in methods:
        lam, loss = best_by_method[m]
        print(f"[{m:13s}] best lambda = {lam:.5g} | best test loss = {loss:.4f}")

    return best_by_method, curves_tr, curves_te



### CROSS VALIDATION DEMO (ALL MODELS) FOR ALL DEGREES AND GAMMAS ###
def cross_validation_demo_all(y, x, degrees, k_fold, lambdas, initial_w, max_iters, gammas, seed=12):


    k_indices = build_k_indices(y, k_fold, seed)

    methods = ["mse_gd","mse_sgd","least_squares","ridge","logistic","reg_logistic"]

    # curves[method][degree][gamma] = (lambdas_array, tr_losses_array, te_losses_array)
    curves = {m: {} for m in methods}

    best_by_method = {m: {"degree": None, "lambda": None, "gamma": None, "test_loss": np.inf} for m in methods}

    for degree in degrees:
        X_tmp = build_poly(x, degree)
        if initial_w is not None and len(initial_w) != X_tmp.shape[1]:
            w0 = np.zeros(X_tmp.shape[1], dtype=float)
        else:
            w0 = initial_w

        for gamma in gammas:
            for m in methods:
                curves[m].setdefault(degree, {})
                curves[m][degree][gamma] = (lambdas.copy(), np.zeros_like(lambdas, dtype=float), np.zeros_like(lambdas, dtype=float))

            # sweep lambdas
            for li, lambda_ in enumerate(lambdas):
        
                sums_tr = {m: 0.0 for m in methods}
                sums_te = {m: 0.0 for m in methods}

                # folds
                for k in range(k_fold):
                    res = cross_validation(
                        y, x, k_indices, k,
                        lambda_, degree,
                        w0, max_iters, gamma
                    )
                    for m in methods:
                        tr_k, te_k = res[m]
                        sums_tr[m] += tr_k
                        sums_te[m] += te_k


                for m in methods:
                    lambdas_arr, tr_arr, te_arr = curves[m][degree][gamma]
                    tr_arr[li] = sums_tr[m] / k_fold
                    te_arr[li] = sums_te[m] / k_fold
                    curves[m][degree][gamma] = (lambdas_arr, tr_arr, te_arr)

  
            for m in methods:
                lambdas_arr, _tr_arr, te_arr = curves[m][degree][gamma]
                bi = int(np.argmin(te_arr))
                best_te = te_arr[bi]
                if best_te < best_by_method[m]["test_loss"]:
                    best_by_method[m] = {
                        "degree": degree,
                        "lambda": float(lambdas_arr[bi]),
                        "gamma": gamma,
                        "test_loss": float(best_te),
                    }

    # summary
    print(f"K-fold = {k_fold}")
    for m in methods:
        info = best_by_method[m]
        print(f"[{m:13s}] best: degree={info['degree']}, lambda={info['lambda']:.5g}, "
              f"gamma={info['gamma']}, test_loss={info['test_loss']:.6f}")

    return best_by_method, curves

