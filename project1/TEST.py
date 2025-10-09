import numpy as np
from cross_validation import cross_validation_demo_all
from data_preprocessing import build_poly
from PCA import *
from plots import plot_method_grid

np.random.seed(0)
n, d = 200, 3
x = np.random.randn(n, d)


y = (x[:, 0] + 0.5*x[:, 1] + 0.2*np.random.randn(n) > 0).astype(int)

# grids — tune to your case
def test(x, y):
    degrees  = [1, 2, 3, 4]
    lambdas  = np.logspace(-4, 2, 9)   # 1e-4 ... 1e2
    gammas   = [0.01, 0.05, 0.1]       
    k_fold   = 5
    max_iters = 1000


    X_max = build_poly(x, max(degrees))
    initial_w = np.zeros(X_max.shape[1])

    best_by_method, curves = cross_validation_demo_all(
        y=y, x=x,
        degrees=degrees,
        k_fold=k_fold,
        lambdas=lambdas,
        initial_w=initial_w,
        max_iters=max_iters,
        gammas=gammas,
        seed=12,
    )

    # Plot a couple of methods
    plot_method_grid("ridge", curves, best_by_method)       # gamma-agnostic
    plot_method_grid("mse_gd", curves, best_by_method)      # multiple gamma lines
    plot_method_grid("reg_logistic", curves, best_by_method)

