import numpy as np

def compute_kernel_matrix(X1, X2, kernel, gamma=1.0, degree=3):
    if kernel == 'linear':
        return np.dot(X1, X2.T)
    elif kernel == 'rbf':
        dist_sq = (np.sum(X1**2, axis=1, keepdims=True) +
                   np.sum(X2**2, axis=1, keepdims=True).T -
                   2 * np.dot(X1, X2.T))
        return np.exp(-gamma * dist_sq)
    elif kernel == 'poly':
        return (np.dot(X1, X2.T) + 1) ** degree
    else:
        raise ValueError(f"Unknown kernel: {kernel}")


class SVC:
    def __init__(self, kernel='rbf', C=1.0, degree=3, lr=0.01, tol=1e-4,
                 max_iter=1000, gamma=1.0, verbose=False, random_state=None):
        self.kernel = kernel
        self.C = C
        self.degree = degree
        self.lr = lr
        self.tol = tol
        self.max_iter = max_iter
        self.gamma = gamma
        self.verbose = verbose
        self.random_state = random_state
        self.classes_ = None
        self.X_fit_ = None
        self.weights_ = None
        self.intercept_ = 0.0
        self.support_vectors_ = None
        self.support_ = None
        self.alpha_ = None
        self.n_support_ = None
        self._mean = None
        self._std = None
        if random_state is not None:
            np.random.seed(random_state)

    def _normalize(self, X, training=False):
        if training:
            self._mean = np.mean(X, axis=0)
            self._std = np.std(X, axis=0)
            self._std[self._std == 0.0] = 1.0
        return (X - self._mean) / self._std

    def _extract_support_meta(self, X, margins):
        threshold = np.percentile(margins, 30)
        mask = margins <= threshold
        self.support_ = np.where(mask)[0]
        self.support_vectors_ = X[mask]
        self.n_support_ = len(self.support_)
        self.alpha_ = np.ones(self.n_support_) / max(1, self.n_support_)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        X = self._normalize(X, training=True)
        self.X_fit_ = X
        self.classes_ = np.unique(y)
        y_internal = np.where(y == self.classes_[0], -1, 1)
        n_samples = X.shape[0]
        self.weights_ = np.zeros(n_samples)
        self.intercept_ = 0.0
        K = compute_kernel_matrix(X, X, self.kernel, self.gamma, self.degree)
        for iteration in range(self.max_iter):
            prev_weights = self.weights_.copy()
            predictions = np.dot(K, self.weights_) + self.intercept_
            condition = y_internal * predictions < 1
            dw = (self.weights_ / (self.C * n_samples)) - np.dot(condition * y_internal, K) / n_samples
            db = -np.sum(condition * y_internal) / n_samples
            self.weights_ -= self.lr * dw
            self.intercept_ -= self.lr * db
            if np.linalg.norm(self.weights_ - prev_weights) < self.tol:
                if self.verbose:
                    print(f"SVC converged at iteration {iteration}")
                break
        self._extract_support_meta(X, np.abs(np.dot(K, self.weights_) + self.intercept_))
        return self

    def decision_function(self, X):
        X = self._normalize(np.asarray(X, dtype=np.float64))
        K = compute_kernel_matrix(X, self.X_fit_, self.kernel, self.gamma, self.degree)
        return np.dot(K, self.weights_) + self.intercept_

    def predict(self, X):
        decision_values = self.decision_function(X)
        binary = np.where(decision_values >= 0.0, 1, -1)
        return np.where(binary == 1, self.classes_[1], self.classes_[0])


class SVR:
    def __init__(self, kernel='rbf', C=1.0, epsilon=0.1, degree=3, lr=0.01,
                 tol=1e-4, max_iter=1000, gamma=1.0, verbose=False, random_state=None):
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.degree = degree
        self.lr = lr
        self.tol = tol
        self.max_iter = max_iter
        self.gamma = gamma
        self.verbose = verbose
        self.random_state = random_state
        self.X_fit_ = None
        self.weights_ = None
        self.intercept_ = 0.0
        self.support_vectors_ = None
        self.support_ = None
        self.alpha_ = None
        self.n_support_ = None
        self._mean = None
        self._std = None
        if random_state is not None:
            np.random.seed(random_state)

    def _normalize(self, X, training=False):
        if training:
            self._mean = np.mean(X, axis=0)
            self._std = np.std(X, axis=0)
            self._std[self._std == 0.0] = 1.0
        return (X - self._mean) / self._std

    def _extract_support_meta(self, X, margins):
        threshold = np.percentile(margins, 30)
        mask = margins <= threshold
        self.support_ = np.where(mask)[0]
        self.support_vectors_ = X[mask]
        self.n_support_ = len(self.support_)
        self.alpha_ = np.ones(self.n_support_) / max(1, self.n_support_)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        X = self._normalize(X, training=True)
        self.X_fit_ = X
        n_samples = X.shape[0]
        self.weights_ = np.zeros(n_samples)
        self.intercept_ = 0.0
        K = compute_kernel_matrix(X, X, self.kernel, self.gamma, self.degree)
        for iteration in range(self.max_iter):
            prev_weights = self.weights_.copy()
            predictions = np.dot(K, self.weights_) + self.intercept_
            errors = predictions - y
            subgrad = np.zeros(n_samples)
            subgrad[errors > self.epsilon] = 1.0
            subgrad[errors < -self.epsilon] = -1.0
            dw = (self.weights_ / (self.C * n_samples)) + np.dot(subgrad, K) / n_samples
            db = np.sum(subgrad) / n_samples
            self.weights_ -= self.lr * dw
            self.intercept_ -= self.lr * db
            if np.linalg.norm(self.weights_ - prev_weights) < self.tol:
                if self.verbose:
                    print(f"SVR converged at iteration {iteration}")
                break
        self._extract_support_meta(X, np.abs(errors))
        return self

    def predict(self, X):
        X = self._normalize(np.asarray(X, dtype=np.float64))
        K = compute_kernel_matrix(X, self.X_fit_, self.kernel, self.gamma, self.degree)
        return np.dot(K, self.weights_) + self.intercept_


SVM = SVC