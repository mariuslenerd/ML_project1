import numpy as np

# ---------- MP-based dimensionality reduction ----------
def mp_dim_reduction(X):
    """
    Standardize → correlation → eigendecompose → keep evals > lambda_plus.
    Returns projected data Z_mp and bookkeeping to compare with PCA.
    """
    T, d = X.shape
    # Standardize columns (correlation model)
    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0, ddof=0)
    Xc = (X - mu) / sd

    # Correlation matrix and eigendecomposition
    S = (Xc.T @ Xc) / T      # correlation (since standardized)
    evals, evecs = np.linalg.eigh(S)  # ascending
    # MP edge (sigma^2=1 in correlation space)
    lam = d / T
    lam_plus = (1 + np.sqrt(lam))**2

    # Keep signal components
    mask = evals > lam_plus
    idx_mp = np.where(mask)[0]             # indices in ascending order
    k_mp = idx_mp.size
    if k_mp == 0:
        # fallback: keep at least 1 component to avoid empty space
        idx_mp = np.array([evals.argmax()])
        k_mp = 1
    W_mp = evecs[:, idx_mp]                # columns = kept eigenvectors
    Z_mp = Xc @ W_mp                       # projected data

    # EVR of correlation eigenvalues (sums to d because corr has unit variances)
    EVR = evals / np.sum(evals)
    # For reporting, sort by descending eigenvalues (common convention)
    order_desc = np.argsort(evals)[::-1]
    evals_desc = evals[order_desc]
    EVR_desc = EVR[order_desc]
    idx_mp_desc = np.flatnonzero(np.in1d(order_desc, idx_mp))

    return Z_mp, k_mp, idx_mp, evals, EVR, evals_desc, EVR_desc, order_desc

# ---------- PCA (EVR threshold) ----------
def PCA_threshold(X, threshold):
    """
    Standard PCA on correlation (standardized data), keeping the
    smallest k s.t. cumulative EVR >= threshold.
    Returns projected data Z_pca and indices of top-k eigenvectors.
    """
    T, d = X.shape
    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0, ddof=0)
    Xc = (X - mu) / sd

    # Correlation matrix via 1/T
    S = (Xc.T @ Xc) / T
    eigvals, eigvecs = np.linalg.eigh(S)       # ascending
    # sort descending for PCA convention
    idx_desc = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx_desc]
    eigvecs = eigvecs[:, idx_desc]

    EVR = eigvals / np.sum(eigvals)
    cumEVR = np.cumsum(EVR)
    k = int(np.searchsorted(cumEVR, threshold) + 1)
    k = min(k, d)

    W = eigvecs[:, :k]
    Z = Xc @ W
    # Map back to original (ascending) indices if you need overlap checks
    idx_col = idx_desc[:k]

    return Z, k, idx_col, eigvals, EVR, cumEVR

## If k_MP is much smaller than k_PCA at your threshold, MP is telling that some of the variance PCA wants to keep looks like noise in high dimensions