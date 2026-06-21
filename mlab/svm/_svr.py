import numpy as np
from ._svm import BaseSVM
from ._svm_utils import compute_kernel_matrix

class SVR(BaseSVM):   
    def __init__(self, kernel: str = 'rbf', C: float = 1.0, epsilon: float = 0.1, degree: int = 3, 
                 lr: float = 0.001, tol: float = 1e-4, max_iter: int = 1000, 
                 gamma: float = 1.0, verbose: bool = False):
        super().__init__(C=C, lr=lr, tol=tol, max_iter=max_iter, 
                         kernel=kernel, degree=degree, gamma=gamma, verbose=verbose)
        self.epsilon = epsilon

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        self.X_fit_ = X
        n_samples = X.shape[0]
        
        self.weights_ = np.zeros(n_samples)
        self.intercept_ = 0.0
        
        K = compute_kernel_matrix(X, X, kernel=self.kernel, gamma=self.gamma, degree=self.degree)
        
        # Subgradient optimizations matching Epsilon-Insensitive loss parameters
        for iteration in range(self.max_iter):
            prev_weights = self.weights_.copy()
            predictions = np.dot(K, self.weights_) + self.intercept_
            errors = predictions - y
            
            # Identify vectors residing external to the targeted allowable error loss tube bounds
            grad_upper = errors > self.epsilon
            grad_lower = errors < -self.epsilon
            
            # Collate dynamic direction indicators
            subgrad_direction = np.zeros(n_samples)
            subgrad_direction[grad_upper] = 1.0
            subgrad_direction[grad_lower] = -1.0
            
            # Analytical gradients
            dw = (self.weights_ / self.C) + np.dot(subgrad_direction, K) / n_samples
            db = np.sum(subgrad_direction) / n_samples
            
            self.weights_ -= self.lr * dw
            self.intercept_ -= self.lr * db
            
            if np.linalg.norm(self.weights_ - prev_weights) < self.tol:
                if self.verbose:
                    print(f"SVR Optimization stabilized at execution iteration: {iteration}")
                break
                
        # Register support metrics for grading evaluators
        margins = np.abs(errors)
        self._extract_support_meta(X, margins)
        return self

    def predict(self, X) -> np.ndarray:
        """Predicts continuous objective values for targeted inputs."""
        return self._compute_decision_values(X)