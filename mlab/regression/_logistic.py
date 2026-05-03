import numpy as np

def sigmoid(z):
    """Computes the sigmoid function."""
    return 1 / (1 + np.exp(-z))

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        # Hyperparameters initialization
        self.learning_rate = learning_rate
        self.max_iterations = n_iterations
        
        self.weights = None
        self.bias = None
        self.cost_history = []

    def fit(self, X, y):
        # Get number of samples and features
        n_samples, n_features = X.shape
        # Initializing weights with zeros. and bias with zero
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.max_iterations):
            # Calculate Gradient and update weights and bias
            y_predict = sigmoid((X @ self.weights) + self.bias)
            derivative_dw = (1/ n_samples) * X.T @ (y_predict - y)
            derivative_db = (1/ n_samples) * np.sum(y_predict - y)

            self.weights -= self.learning_rate * derivative_dw
            self.bias -= self.learning_rate * derivative_db

            # Store cost every iteration to check for convergence and monitor it
            cost = self._compute_cost(y, y_predict)
            self.cost_history.append(cost)      
        return self

    def predict(self, X):
        """class 1 if probability more than or equal to 0.5 else class 0"""
        return (self.predict_proba(X) >= 0.5).astype(int)

    def predict_proba(self, X):
        return sigmoid((X @ self.weights) + self.bias)
       
    
    def _compute_cost(self, y_true, y_pred):
        # Binary cross-entropy cost function
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1- epsilon)
        cost = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1- y_pred))
        return cost

class SGDClassifier:
    def __init__(self, learning_rate=0.01, n_iterations=1000, batch_size=32):
        # Hyperparameters
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size

        # Model parameters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """Train the model using stochastic gradient descent."""
        # Get number of samples and features
        n_samples, n_features = X.shape

        # Initializing weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iterations):
            # shuffling data at each iteration to improve convergence speed
            indices = np.random.permutation(n_samples)
            X_sh, y_sh = X[indices], y[indices]
            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size

                xi = X_sh[start:end]
                yi = y_sh[start:end]
                
                # Compute predictions and error for the current batch
                y_pred = sigmoid((xi @ self.weights) + self.bias)
                error = y_pred - yi
                
                # Updating weights and bias using the error and learning rate
                m = len(yi)
                self.weights -= self.learning_rate * (1/m) * (xi.T @ error)
                self.bias -= self.learning_rate * (1/m) * np.sum(error)

    def predict(self, X):
        """Predict class labels for samples in X."""

        # Compute probabilities and convert to class labels
        probs = sigmoid((X @ self.weights) + self.bias)
        # Class 1 if probability more than or equal to 0.5 else class 0
        return (probs >= 0.5).astype(int)
