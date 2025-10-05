import numpy as np

def cross_validation_visualization(lambds, rmse_tr, rmse_te, title_suffix=""):
    import matplotlib.pyplot as plt
    plt.semilogx(lambds, rmse_tr, marker=".", color="b", label="train error")
    plt.semilogx(lambds, rmse_te, marker=".", color="r", label="test error")
    plt.xlabel("lambda")
    plt.ylabel("loss")
    plt.title(f"cross validation {title_suffix}".strip())
    plt.legend(loc=2)
    plt.grid(True)
    plt.show()

def plot_all_methods(lambdas, curves_tr, curves_te):
    """
    curves_tr/curves_te: dict method -> list of fold-averaged losses over lambdas
    """
    for method in curves_tr.keys():
        cross_validation_visualization(lambdas, np.array(curves_tr[method]), np.array(curves_te[method]), title_suffix=f"({method})")

def plot_method_grid(method, curves, best_info):
    """
    Plot, for a single method:
      - Fix degree at that method's best degree
      - Plot test loss vs λ, one line per γ (if applicable)
    """
    import matplotlib.pyplot as plt

    degree_best = best_info[method]["degree"]
    entries = curves[method][degree_best]  # dict gamma -> (lambdas, tr, te)

    plt.figure()
    for gamma, (lambdas_arr, _tr_arr, te_arr) in entries.items():
        label = f"gamma={gamma}" if gamma is not None else "gamma=None"
        plt.semilogx(lambdas_arr, te_arr, marker=".", label=label)

    plt.xlabel("lambda")
    plt.ylabel("test loss")
    plt.title(f"{method} | best degree={degree_best}")
    plt.legend()
    plt.grid(True)
    plt.show()

