import numpy as np
from implemented_functions import *
from data_preprocessing import build_poly
import matplotlib.pyplot as plt

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


###############################################################################
# Helpers
###############################################################################
def init_w_for_degree(initial_w, n_features):
    """Return an initial w of the right length for this polynomial degree."""
    if len(initial_w) == n_features:
        return initial_w
    return np.zeros(n_features, dtype=float)

def cross_validation_single(
    y, x, k_indices, k, lambda_, degree,initial_w, max_iters, gamma, method
):
    """
    Train ONE method on fold k and return (train_loss, test_loss) for that method.
    """

    te_idx = k_indices[k]
    tr_idx = np.hstack([k_indices[i] for i in range(k_indices.shape[0]) if i != k])

    x_tr_k = x[tr_idx]
    y_tr_k = y[tr_idx]
    x_te_k = x[te_idx]
    y_te_k = y[te_idx]

    x_tr_poly = build_poly(x_tr_k, degree, interactions=False)
    x_te_poly = build_poly(x_te_k, degree,interactions=False)
    # --- regression-style losses are RMSE, logistic-style are log-loss ---
    if method == "mse_gd":
        w, _ = mean_squared_error_gd(y_tr_k, x_tr_poly, initial_w, max_iters, gamma)
        tr_loss = compute_rmse(y_tr_k, x_tr_poly, w)
        te_loss = compute_rmse(y_te_k, x_te_poly, w)

    elif method == "mse_sgd":
        w, _ = mean_squared_error_sgd(y_tr_k, x_tr_poly, initial_w, max_iters, gamma)
        tr_loss = compute_rmse(y_tr_k, x_tr_poly, w)
        te_loss = compute_rmse(y_te_k, x_te_poly, w)

    elif method == "least_squares":
        w, _ = least_squares(y_tr_k, x_tr_poly)
        tr_loss = compute_rmse(y_tr_k, x_tr_poly, w)
        te_loss = compute_rmse(y_te_k, x_te_poly, w)

    elif method == "ridge":
        w, _ = ridge_regression(y_tr_k, x_tr_poly, lambda_)
        tr_loss = compute_rmse(y_tr_k, x_tr_poly, w)
        te_loss = compute_rmse(y_te_k, x_te_poly, w)

    elif method == "logistic":
        w, _ = logistic_regression(y_tr_k, x_tr_poly, initial_w, max_iters, gamma)
        tr_loss = compute_logistic_loss(y_tr_k, x_tr_poly, w)
        te_loss = compute_logistic_loss(y_te_k, x_te_poly, w)

    elif method == "reg_logistic":
        w, _ = reg_logistic_regression(
            y_tr_k, x_tr_poly, lambda_, initial_w, max_iters, gamma
        )
        tr_loss = compute_reg_logistic_loss(y_tr_k, x_tr_poly, w, lambda_)
        te_loss = compute_reg_logistic_loss(y_te_k, x_te_poly, w, lambda_)

    elif method == "reg_lasso_logistic":
        # subgradient L1 solver
        w, _ = reg_logistic_lasso_subgradient(
            y_tr_k, x_tr_poly, lambda_, initial_w, max_iters, gamma
        )
        tr_loss = compute_reg_logistic_loss_l1(y_tr_k, x_tr_poly, w, lambda_)
        te_loss = compute_reg_logistic_loss_l1(y_te_k, x_te_poly, w, lambda_)

    else:
        raise ValueError(f"Unknown method {method}")

    return tr_loss, te_loss


###############################################################################
# MAIN CV DRIVER
###############################################################################

def cross_validation_demo_all(
    y,
    x,
    degrees,
    k_fold,
    lambdas,
    initial_w,
    max_iters,
    gammas,
    seed=12,
    verbose=True,
):
    """
    Run K-fold CV for all methods, but:
    - Only sweep gamma for methods that depend on gamma.
    - Only sweep lambda for methods that depend on lambda.
    - Sweep both for methods that depend on both.
    - No sweep if method has neither.

    Also prints live progress and returns:
        best_by_method[method] = {
            "degree", "lambda", "gamma", "test_loss"
        }

    curves[method][degree] has this shape depending on hyperparams:
        if method needs neither:
            {
              "train": float,
              "test": float
            }
        if only lambda:
            {
              "lambdas": np.array([...]),
              "train":  np.array([...]),
              "test":   np.array([...])
            }
        if only gamma:
            {
              "gammas": np.array([...]),
              "train":  np.array([...]),
              "test":   np.array([...])
            }
        if both:
            {
              "lambdas": np.array([...]),
              "gammas":  np.array([...]),
              "train":   np.array[[len(gammas), len(lambdas)]],
              "test":    np.array[[len(gammas), len(lambdas)]],
            }
    """

    # define which hyperparams each method actually uses
    method_specs = {
        "mse_gd":               {"need_lambda": False, "need_gamma": True},
        "mse_sgd":              {"need_lambda": False, "need_gamma": True},
        "least_squares":        {"need_lambda": False, "need_gamma": False},
        "ridge":                {"need_lambda": True,  "need_gamma": False},
        "logistic":             {"need_lambda": False, "need_gamma": True},
        "reg_logistic":         {"need_lambda": True,  "need_gamma": True},
        "reg_lasso_logistic":   {"need_lambda": True,  "need_gamma": True},
    }

    methods = list(method_specs.keys())

    # init storage
    curves = {m: {} for m in methods}
    best_by_method = {
        m: {"degree": None, "lambda": None, "gamma": None, "test_loss": np.inf}
        for m in methods
    }

    # build K-fold indices
    k_indices = build_k_indices(y, k_fold, seed)

    # precompute total steps for pretty progress info
    def n_combos(spec):
        need_l, need_g = spec["need_lambda"], spec["need_gamma"]
        if need_l and need_g:
            return len(lambdas) * len(gammas)
        elif need_l:
            return len(lambdas)
        elif need_g:
            return len(gammas)
        else:
            return 1

    total_steps = sum(
        n_combos(method_specs[m]) * len(degrees)
        for m in methods
    )
    step = 0

    # loop over polynomial degrees
    for degree in degrees:
        # figure out correct w0 length for this degree
        X_tmp = build_poly(x, degree)
        w0 = init_w_for_degree(initial_w, X_tmp.shape[1])

        for m in methods:
            spec = method_specs[m]
            need_l, need_g = spec["need_lambda"], spec["need_gamma"]

            # prepare curve containers for this degree+method
            if need_l and need_g:
                curves[m][degree] = {
                    "lambdas": np.array(lambdas, dtype=float),
                    "gammas": np.array(gammas, dtype=float),
                    "train": np.zeros((len(gammas), len(lambdas)), dtype=float),
                    "test":  np.zeros((len(gammas), len(lambdas)), dtype=float),
                }
            elif need_l:
                curves[m][degree] = {
                    "lambdas": np.array(lambdas, dtype=float),
                    "train": np.zeros(len(lambdas), dtype=float),
                    "test":  np.zeros(len(lambdas), dtype=float),
                }
            elif need_g:
                curves[m][degree] = {
                    "gammas": np.array(gammas, dtype=float),
                    "train": np.zeros(len(gammas), dtype=float),
                    "test":  np.zeros(len(gammas), dtype=float),
                }
            else:
                curves[m][degree] = {
                    "train": 0.0,
                    "test":  0.0,
                }

            # now actually run CV sweeps
            if need_l and need_g:
                # double loop
                for gi, gamma in enumerate(gammas):
                    for li, lambda_ in enumerate(lambdas):

                        # progress print
                        step += 1
                        if verbose:
                            print(
                                f"[{step}/{total_steps}] method={m}, degree={degree}, "
                                f"gamma={gamma:.3g}, lambda={lambda_:.3g}"
                            )

                        # accumulate fold results
                        tr_sum = 0.0
                        te_sum = 0.0
                        for k in range(k_fold):
                            tr_k, te_k = cross_validation_single(
                                y, x, k_indices, k,
                                lambda_, degree,
                                w0, max_iters, gamma,
                                m
                            )
                            tr_sum += tr_k
                            te_sum += te_k

                        tr_avg = tr_sum / k_fold
                        te_avg = te_sum / k_fold

                        curves[m][degree]["train"][gi, li] = tr_avg
                        curves[m][degree]["test"][gi,  li] = te_avg

                        # track best
                        if te_avg < best_by_method[m]["test_loss"]:
                            best_by_method[m] = {
                                "degree": degree,
                                "lambda": float(lambda_),
                                "gamma": float(gamma),
                                "test_loss": float(te_avg),
                            }

            elif need_l:
                # sweep only lambdas
                for li, lambda_ in enumerate(lambdas):

                    step += 1
                    if verbose:
                        print(
                            f"[{step}/{total_steps}] method={m}, degree={degree}, "
                            f"lambda={lambda_:.3g}"
                        )

                    tr_sum = 0.0
                    te_sum = 0.0
                    for k in range(k_fold):
                        tr_k, te_k = cross_validation_single(
                            y, x, k_indices, k,
                            lambda_, degree,
                            w0, max_iters, gamma=None,
                            method=m
                        )
                        tr_sum += tr_k
                        te_sum += te_k

                    tr_avg = tr_sum / k_fold
                    te_avg = te_sum / k_fold

                    curves[m][degree]["train"][li] = tr_avg
                    curves[m][degree]["test"][li]  = te_avg

                    if te_avg < best_by_method[m]["test_loss"]:
                        best_by_method[m] = {
                            "degree": degree,
                            "lambda": float(lambda_),
                            "gamma": None,
                            "test_loss": float(te_avg),
                        }

            elif need_g:
                # sweep only gammas
                for gi, gamma in enumerate(gammas):

                    step += 1
                    if verbose:
                        print(
                            f"[{step}/{total_steps}] method={m}, degree={degree}, "
                            f"gamma={gamma:.3g}"
                        )

                    tr_sum = 0.0
                    te_sum = 0.0
                    for k in range(k_fold):
                        tr_k, te_k = cross_validation_single(
                            y, x, k_indices, k,
                            lambda_=0.0,  # ignored by these methods
                            degree=degree,
                            initial_w=w0,
                            max_iters=max_iters,
                            gamma=gamma,
                            method=m
                        )
                        tr_sum += tr_k
                        te_sum += te_k

                    tr_avg = tr_sum / k_fold
                    te_avg = te_sum / k_fold

                    curves[m][degree]["train"][gi] = tr_avg
                    curves[m][degree]["test"][gi]  = te_avg

                    if te_avg < best_by_method[m]["test_loss"]:
                        best_by_method[m] = {
                            "degree": degree,
                            "lambda": None,
                            "gamma": float(gamma),
                            "test_loss": float(te_avg),
                        }

            else:
                # no hyperparams at all -> single fit
                step += 1
                if verbose:
                    print(f"[{step}/{total_steps}] method={m}, degree={degree} (no hp)")

                tr_sum = 0.0
                te_sum = 0.0
                for k in range(k_fold):
                    tr_k, te_k = cross_validation_single(
                        y, x, k_indices, k,
                        lambda_=0.0,
                        degree=degree,
                        initial_w=w0,
                        max_iters=max_iters,
                        gamma=0.0,
                        method=m
                    )
                    tr_sum += tr_k
                    te_sum += te_k
                tr_avg = tr_sum / k_fold
                te_avg = te_sum / k_fold

                curves[m][degree]["train"] = tr_avg
                curves[m][degree]["test"]  = te_avg

                if te_avg < best_by_method[m]["test_loss"]:
                    best_by_method[m] = {
                        "degree": degree,
                        "lambda": None,
                        "gamma": None,
                        "test_loss": float(te_avg),
                    }

    # SUMMARY PRINT
    print("\n=== CV SUMMARY ===")
    print(f"K-fold = {k_fold}")
    for m in methods:
        info = best_by_method[m]
        lam  = "None" if info["lambda"] is None else f"{info['lambda']:.5g}"
        gam  = "None" if info["gamma"]  is None else f"{info['gamma']:.5g}"
        print(
            f"[{m:18s}] "
            f"best: degree={info['degree']}, "
            f"lambda={lam}, gamma={gam}, "
            f"test_loss={info['test_loss']:.6f}"
        )

    return best_by_method, curves


###############################################################################
# PLOTTING
###############################################################################

def plot_cv_results(curves, method, degree, show_train=True, log_x=True):
    """
    Nice plotting for a single method @ one degree.

    - if method only uses lambda: plot test/train vs lambda
    - if method only uses gamma:  plot test/train vs gamma
    - if method uses both:       show heatmap (gamma vs lambda)
    - if method uses neither:    just print the scalar

    Parameters
    ----------
    curves : output from cross_validation_demo_all
    method : str, one of keys of `curves`
    degree : the degree you want to visualize
    show_train : bool, also plot training curve
    log_x : bool, use log-scale on x-axis for 1D sweeps
    """

    data = curves[method][degree]

    # figure out what hyperparams exist in this curve
    has_lambda = "lambdas" in data
    has_gamma  = "gammas"  in data

    if not has_lambda and not has_gamma:
        # trivial case
        test_loss = data["test"]
        train_loss = data["train"]
        print(
            f"{method} (degree={degree}): "
            f"train={train_loss:.4f}, test={test_loss:.4f}"
        )
        return

    if has_lambda and not has_gamma:
        # 1D sweep over lambda
        lambdas_arr = data["lambdas"]
        te_arr = data["test"]
        tr_arr = data["train"]

        plt.figure()
        if show_train:
            plt.plot(lambdas_arr, tr_arr, marker="o", label="train")
        plt.plot(lambdas_arr, te_arr, marker="s", label="test")
        plt.xlabel("lambda")
        plt.ylabel("loss")
        plt.title(f"{method} | degree={degree}")
        plt.grid(True)
        plt.legend()
        if log_x:
            plt.xscale("log")
        plt.show()
        return

    if has_gamma and not has_lambda:
        # 1D sweep over gamma
        gammas_arr = data["gammas"]
        te_arr = data["test"]
        tr_arr = data["train"]

        plt.figure()
        if show_train:
            plt.plot(gammas_arr, tr_arr, marker="o", label="train")
        plt.plot(gammas_arr, te_arr, marker="s", label="test")
        plt.xlabel("gamma")
        plt.ylabel("loss")
        plt.title(f"{method} | degree={degree}")
        plt.grid(True)
        plt.legend()
        if log_x:
            plt.xscale("log")
        plt.show()
        return

    if has_gamma and has_lambda:
        # 2D grid -> heatmap of test loss
        gammas_arr = data["gammas"]
        lambdas_arr = data["lambdas"]
        test_grid = data["test"]  # shape [len(gammas), len(lambdas)]

        plt.figure()
        # imshow expects [row, col] so row -> gamma index, col -> lambda index
        im = plt.imshow(
            test_grid,
            origin="lower",
            aspect="auto",
        )
        plt.colorbar(im, label="test loss")

        # ticks: try to show a few readable ticks
        # we'll choose ~5 ticks max for clarity
        max_ticks = 5
        g_idx = np.linspace(0, len(gammas_arr) - 1, num=min(len(gammas_arr), max_ticks)).astype(int)
        l_idx = np.linspace(0, len(lambdas_arr) - 1, num=min(len(lambdas_arr), max_ticks)).astype(int)

        plt.xticks(l_idx, [f"{lambdas_arr[i]:.2g}" for i in l_idx], rotation=45)
        plt.yticks(g_idx, [f"{gammas_arr[i]:.2g}" for i in g_idx])

        plt.xlabel("lambda")
        plt.ylabel("gamma")
        plt.title(f"{method} | degree={degree} | test loss heatmap")
        plt.tight_layout()
        plt.show()
        return



