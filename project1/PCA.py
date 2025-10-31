import numpy as np

# ---------- MP-based dimensionality reduction ----------
def mp_dim_reduction(X_tr,X_te, X_te_fin):
    """
    Standardize, correlation,  eigendecompose, keep evals > lambda_plus.
    """
    T, d = X_tr.shape

    # Standardize 
    mu_tr = np.mean(X_tr, axis=0)
    sd_tr = np.std(X_tr, axis=0, ddof=1)
    zero_var = (sd_tr == 0) | ~np.isfinite(sd_tr)
    idx_kept = np.where(~zero_var)[0]

    Xc_tr = (X_tr[:,idx_kept] - mu_tr[idx_kept]) / sd_tr[idx_kept]
    Xc_te = (X_te[:,idx_kept] - mu_tr[idx_kept]) / sd_tr[idx_kept]
    Xc_te_fin = (X_te_fin[:,idx_kept] - mu_tr[idx_kept]) / sd_tr[idx_kept]

    # Correlation matrix and eigendecomposition
    S = (Xc_tr.T @ Xc_tr) / T      # correlation
    evals, evecs = np.linalg.eigh(S) 
    # MP edge (sigma^2=1 in correlation)
    lam = d / T
    lam_plus = (1 + np.sqrt(lam))**2

    # Keep signal components
    mask = evals > lam_plus
    idx_mp = np.where(mask)[0]            
    k_mp = idx_mp.size
    if k_mp == 0:
        # fallback: keep at least 1 component to avoid empty space
        idx_mp = np.array([evals.argmax()])
        k_mp = 1
    W_mp = evecs[:, idx_mp]                
    Z_mp = Xc_tr @ W_mp # projected train data
    Z_te_mp = Xc_te @ W_mp # projected test data
    Z_te_fin = Xc_te_fin @ W_mp
    EVR = evals / np.sum(evals)
    order_desc = np.argsort(evals)[::-1]
    evals_desc = evals[order_desc]
    EVR_desc = EVR[order_desc]

    return Z_mp,Z_te_mp,Z_te_fin

# ---------- PCA (EVR threshold) ----------
def PCA_threshold(X_tr,X_te,X_te_fin, threshold):
    """
    Standard PCA on correlation (standardized data), keeping the
    smallest k s.t. cumulative EVR >= threshold.
    Returns projected data Z_pca and indices of top-k eigenvectors.
    """
    T, d = X_tr.shape

    mu_tr = np.mean(X_tr, axis=0)
    sd_tr = np.std(X_tr, axis=0, ddof=1)
    zero_var = (sd_tr == 0) | ~np.isfinite(sd_tr)
    idx_kept = np.where(~zero_var)[0]

    Xc_tr = (X_tr[:,idx_kept] - mu_tr[idx_kept]) / sd_tr[idx_kept]
    Xc_te = (X_te[:,idx_kept] - mu_tr[idx_kept]) / sd_tr[idx_kept]
    Xc_te_fin = (X_te_fin[:,idx_kept] - mu_tr[idx_kept]) / sd_tr[idx_kept]
    # Correlation matrix 
    S = (Xc_tr.T @ Xc_tr) / T
    eigvals, eigvecs = np.linalg.eigh(S)     

    idx_desc = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx_desc]
    eigvecs = eigvecs[:, idx_desc]

    EVR = eigvals / np.sum(eigvals)
    cumEVR = np.cumsum(EVR)
    k = int(np.searchsorted(cumEVR, threshold) + 1)
    k = min(k, d)

    W = eigvecs[:, :k]
    Z_train= Xc_tr @ W # projected train data
    Z_test = Xc_te @ W # projected test data
    Z_te_fin = Xc_te_fin @ W

    idx_col = idx_desc[:k]

    return Z_train,Z_test,Z_te_fin

