import numpy as np

def compute_kernel_matrix(X1: np.ndarray, X2: np.ndarray, kernel: str, gamma: float = 1.0, degree: int = 3) -> np.ndarray:
    """
    Computes the kernel matrix between two sets of samples X1 and X2.
    """
    if kernel == 'linear':
        return np.dot(X1, X2.T)
        
    elif kernel == 'rbf':
        # Vectorized pairwise squared Euclidean distances: ||x - y||^2 = ||x||^2 + ||y||^2 - 2x.y^T
        dist_sq = (np.sum(X1**2, axis=1, keepdims=True) + 
                   np.sum(X2**2, axis=1, keepdims=True).T - 
                   2 * np.dot(X1, X2.T))
        return np.exp(-gamma * dist_sq)
        
    elif kernel == 'poly':
        return (np.dot(X1, X2.T) + 1) ** degree
        
    else:
        raise ValueError(f"Unknown kernel function: {kernel}")