import numpy as np

class GaussianNaiveBayes:
    def __init__(self):
        self.classes_ = None    # unique class labels (set after fit)
        self.priors_ = None     # prior probabilities per class
        self.mean_ = None       # mean of each feature per class
        self.variance_ = None   # variance of each feature per class

    def fit(self, X, y):
        """
        Train the model by computing class priors, means, and variances.

        Args:
            X: numpy array of shape (n_samples, n_features) - continuous features
            y: numpy array of shape (n_samples,) - class labels
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        
        # Handling edge case empty input
        if X.size == 0 or y.size == 0:
            return self
        
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        
        # Initializing parameters
        self.priors_ = np.zeros(n_classes, dtype=np.float64)
        self.mean_ = np.zeros((n_classes, n_features), dtype=np.float64)
        self.variance_ = np.zeros((n_classes, n_features), dtype=np.float64)
        
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.priors_[idx] = X_c.shape[0] / n_samples
            self.mean_[idx, :] = np.mean(X_c, axis=0)
            # Added a tiny epsilon to prevent division by zero
            self.variance_[idx, :] = np.var(X_c, axis=0) + 1e-9

    def _get_log_posteriors(self, X):
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        log_posteriors = np.zeros((n_samples, n_classes), dtype=np.float64)
        
        for idx in range(n_classes):
            mean = self.mean_[idx]
            var = self.variance_[idx]
            prior = self.priors_[idx]
            
            # Log Likelihood formula
            term1 = -0.5 * np.sum(np.log(2 * np.pi * var))
            term2 = -0.5 * np.sum(((X - mean) ** 2) / var, axis=1)
            log_likelihood = term1 + term2
            log_posteriors[:, idx] = log_likelihood + np.log(prior)
            
        return log_posteriors
    
    def predict(self, X):
        """
        Predict class labels for the given input.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples,) with predicted class labels
        """
        # when data is empty then return empty array
        X = np.array(X, dtype=np.float64)
        if X.size == 0:
            return np.array([])
        
        log_posteriors = self._get_log_posteriors(X)
        class_indices = np.argmax(log_posteriors, axis=1)
        return self.classes_[class_indices]

    def predict_proba(self, X):
        """
        Predict class probabilities. (Optional but recommended)

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples, n_classes)
        """
        # when data is empty then return empty array
        X = np.array(X, dtype=np.float64)
        if X.size == 0:
            return np.array([[]])
        
        log_posteriors = self._get_log_posteriors(X)
        
        # Log-Sum-Exp stabilization Trick
        max_log = np.max(log_posteriors, axis=1, keepdims=True)
        exp_posteriors = np.exp(log_posteriors - max_log)
        probabilities = exp_posteriors / np.sum(exp_posteriors, axis=1, keepdims=True)
        
        return probabilities


class MultinomialNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha          # Laplace smoothing parameter
        self.classes_ = None
        self.class_log_prior_ = None
        self.feature_log_prob_ = None

    def fit(self, X, y):
        """
        Train the model by computing class priors and feature likelihoods.

        Args:
            X: numpy array of shape (n_samples, n_features) - non-negative count features
            y: numpy array of shape (n_samples,) - class labels
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        
        # Handling edge case empty input
        if X.size == 0 or y.size == 0:
            return self
        
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        
        # Initializing parameters
        self.class_log_prior_ = np.zeros(n_classes, dtype=np.float64)
        self.feature_log_prob_ = np.zeros((n_classes, n_features), dtype=np.float64)
        
        class_counts = np.array([np.sum(y == c) for c in self.classes_])
        self.class_log_prior_ = np.log(class_counts / n_samples)
        
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            feature_counts = np.sum(X_c, axis=0)
            
            # Laplace smoothing
            smoothed_counts = feature_counts + self.alpha
            smoothed_total = np.sum(feature_counts) + (self.alpha * n_features)
            
            self.feature_log_prob_[idx, :] = np.log(smoothed_counts / smoothed_total)

    def _get_log_posteriors(self, X):
        X = np.asarray(X, dtype=np.float64)
        return X @ self.feature_log_prob_.T + self.class_log_prior_

    def predict(self, X):
        """
        Predict class labels for the given input.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples,) with predicted class labels
        """
        log_posteriors = self._get_log_posteriors(X)
        class_indices = np.argmax(log_posteriors, axis=1)
        return self.classes_[class_indices]

    def predict_proba(self, X):
        """
        Predict class probabilities. (Optional but recommended)

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples, n_classes)
        """
        log_posteriors = self._get_log_posteriors(X)
        
        # Log-Sum-Exp stabilization Trick
        max_log = np.max(log_posteriors, axis=1, keepdims=True)
        exp_posteriors = np.exp(log_posteriors - max_log)
        probabilities = exp_posteriors / np.sum(exp_posteriors, axis=1, keepdims=True)
        
        return probabilities