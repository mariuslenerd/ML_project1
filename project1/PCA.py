import numpy as np

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