import numpy as np
from ._svm import BaseSVM
from ._svm_utils import compute_kernel_matrix

class SVC(BaseSVM):
    def __init__(self, kernel: str = 'rbf', C: float = 1.0, degree: int = 3, 
                 lr: float = 0.01, tol: float = 1e-4, max_iter: int = 1000, 
                 gamma: float = 1.0, verbose: bool = False, random_state: int = None):
        super().__init__(C=C, lr=lr, tol=tol, max_iter=max_iter, 
                         kernel=kernel, degree=degree, gamma=gamma, 
                         verbose=verbose, random_state=random_state)
        self.classes_ = None
        
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        
        X = self._normalize_features(X, training=True)

        self.X_fit_ = X
        self.classes_ = np.unique(y)
        
        # Map dynamic unique user class labels into explicit internal operational values {-1, 1}
        y_internal = np.where(y == self.classes_[0], -1, 1)
        
        n_samples = X.shape[0]
        self.weights_ = np.zeros(n_samples)
        self.intercept_ = 0.0
        
        # Precompute internal target Kernel configuration matrix
        K = compute_kernel_matrix(X, X, kernel=self.kernel, gamma=self.gamma, degree=self.degree)
        
        # Subgradient Descent updates for Soft-Margin hinge constraints
        for iteration in range(self.max_iter):
            prev_weights = self.weights_.copy()
            
            # Vectorized decision value generation loop
            predictions = np.dot(K, self.weights_) + self.intercept_
            condition = y_internal * predictions < 1
            
            # Loss and updates computation 
            # Subgradient of Hinge Loss w.r.t weights coefficient is -y * K_row
            dw = (self.weights_ / (self.C * n_samples)) - np.dot(condition * y_internal, K) / n_samples
            db = -np.sum(condition * y_internal) / n_samples
            
            self.weights_ -= self.lr * dw
            self.intercept_ -= self.lr * db
            
            # Check weight vector optimization delta delta stability tolerance parameters
            if np.linalg.norm(self.weights_ - prev_weights) < self.tol:
                if self.verbose:
                    print(f"SVC Optimization stabilized at execution iteration: {iteration}")
                break
                
        # Generate evaluative tracking attributes
        margins = np.abs(np.dot(K, self.weights_) + self.intercept_)
        self._extract_support_meta(X, margins)
        return self

    def decision_function(self, X) -> np.ndarray:
        """Computes the distances of target input points mapping to the separation hyperplanes."""
        X = np.asarray(X, dtype=np.float64)
        X_scaled = self._normalize_features(X, training=False)
        return self._compute_decision_values(X_scaled)

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        X_scaled = self._normalize_features(X, training=False)
        decision_values = self._compute_decision_values(X_scaled)
        binary_predictions = np.where(decision_values >= 0.0, 1, -1)
        return np.where(binary_predictions == 1, self.classes_[1], self.classes_[0])