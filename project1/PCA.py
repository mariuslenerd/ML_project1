import numpy as np
import matplotlib.pyplot as plt

def PCA(X,threshold):

    """
    Perform Principal Component Analysis (PCA) on a dataset.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data matrix where rows are samples and columns are features.
    
    threshold : float in (0,1]
        The minimum cumulative explained variance ratio required.
        Example: threshold=0.95 will keep enough principal components
        to explain at least 95% of the variance.

    Returns
    -------
    Z : ndarray of shape (n_samples, k)
        The transformed data projected onto the selected principal components.
    
    k : int
        The number of principal components retained to reach the threshold.
    
    idx_col : ndarray of shape (k,)
        Indices of the selected components
    """

    Xc = (X - np.mean(X, axis= 0))/ np.std(X,axis=0)
    cov = np.cov(Xc, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:,idx]
    EVR = eigvals / np.sum(eigvals)
    k = np.searchsorted(np.cumsum(EVR), threshold) + 1 
    W = eigvecs[:,:k]
    idx_col = idx[:k]
    Z = Xc @ W
    return Z,k,idx_col


def ridge_regr(signals: np.ndarray,
                  labels: np.ndarray,
                  shrinkage_list: np.ndarray):
    """
    Regression is
    beta = (zI + S'S/t)^{-1}S'y/t = S' (zI+SS'/t)^{-1}y/t
    Inverting matrices is costly, so we use eigenvalue decomposition:
    (zI+A)^{-1} = U (zI+D)^{-1} U' where UDU' = A is eigenvalue decomposition,
    and we use the fact that D @ B = (diag(D) * B) for diagonal D, which saves a lot of compute cost
    :param signals: S
    :param labels: y
    :param future_signals: out of sample y
    :param shrinkage_list: list of ridge parameters
    :return:
    """
    signals_c = (signals - np.mean(signals, axis= 0))/ np.std(signals,axis=0)
    t_ = signals_c.shape[0]
    p_ = signals_c.shape[1]
    if p_ < t_:
        # this is standard regression
        eigenvalues, eigenvectors = np.linalg.eigh(signals_c.T @ signals_c / t_)
        means = signals_c.T @ labels.reshape(-1, 1) / t_
        multiplied = eigenvectors.T @ means

        # now we deal with a whole grid of ridge penalties
        intermed = np.concatenate([(1 / (eigenvalues.reshape(-1, 1) + z)) * multiplied for z in shrinkage_list],
                                  axis=1)
        betas = eigenvectors @ intermed
    else:
        # this is the weird over-parametrized regime
        eigenvalues, eigenvectors = np.linalg.eigh(signals_c @ signals_c.T / t_)
        means = labels.reshape(-1, 1) / t_
        multiplied = eigenvectors.T @ means

        # now we deal with a whole grid of ridge penalties
        intermed = np.concatenate([(1 / (eigenvalues.reshape(-1, 1) + z)) * multiplied for z in shrinkage_list],
                                  axis=1)
        tmp = eigenvectors.T @ signals_c
        betas = tmp.T @ intermed
    return betas

def plot_ridge_betas(alphas, betas, feature_names):
    n_alphas, n_features = betas.shape

    # 1) Coefficient paths: coefficient value vs log(alpha)
    plt.figure(figsize=(8, 6))
    for j in range(n_features):
        plt.plot(alphas, betas[:, j], label=feature_names[j])
    plt.xscale("log")
    plt.xlabel("Alpha (log scale)")
    plt.ylabel("Coefficient")
    plt.title("Ridge Coefficient Paths")
    plt.legend()
    plt.grid(True)

    # 2) Heatmap of coefficients (alphas x features)
    plt.figure(figsize=(max(6, n_features*0.7), max(4, n_alphas*0.15)))
    plt.imshow(betas, aspect="auto", origin="lower")
    plt.colorbar(label="Coefficient")
    plt.yticks(
        ticks=np.linspace(0, n_alphas-1, min(n_alphas, 10), dtype=int),
        labels=[f"{alphas[i]:.3g}" for i in np.linspace(0, n_alphas-1, min(n_alphas, 10), dtype=int)]
    )
    plt.xticks(ticks=np.arange(n_features), labels=feature_names, rotation=45, ha="right")
    plt.xlabel("Features")
    plt.ylabel("Alphas")
    plt.title("Ridge Coefficients Heatmap")
    
    # 3) Bar charts at 3 representative alphas 
    chosen_idx = sorted(set([0, (n_alphas-1)//2, n_alphas-1]))
    for idx in chosen_idx:
        plt.figure(figsize=(8, 5))
        vals = betas[idx]
        order = np.argsort(np.abs(vals))[::-1]   # rank by magnitude
        plt.bar(np.arange(n_features), vals[order])
        plt.xticks(ticks=np.arange(n_features), labels=[feature_names[i] for i in order], rotation=45, ha="right")
        plt.xlabel("Features (sorted by |beta|)")
        plt.ylabel("Coefficient")
        plt.title(f"Ridge Coefficients at alpha={alphas[idx]:.3g}")
        plt.grid(True, axis="y")

    plt.show()