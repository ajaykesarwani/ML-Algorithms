import numpy as np
class LinearRegression:
    def __init__(self):
        self.weights_ = None  # learned coefficients (set after fit)
        self.bias_ = None

    def fit(self, X, y):
        """
        Train the model on the given data.

        Args:
            X: numpy array of shape (n_samples, n_features)
            y: numpy array of shape (n_samples,)
        """
        # # Adding bias column to the input X
        X_expand = np.concatenate((np.ones((X.shape[0], 1)), np.array(X)), axis=1)
        X_expand_T = X_expand.T
        # w = (X^T X)^{-1} X^T y
        w = np.linalg.inv(X_expand_T @ X_expand) @ X_expand_T @ y
        self.bias_ = w[0]
        self.weights_ = w[1:]


    def predict(self, X):
        """
        Predict target values for the given input.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples,)
        """
        # Adding bias column to the input X
        # feature_expand = np.concatenate((np.ones((X.shape[0], 1)), np.array(X)), axis=1)
        return (X @ self.weights_ + self.bias_).flatten()

class SGDRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, batch_size=32):
        # Hyperparameter initialization
        self.lr = learning_rate
        self.n_iters = n_iterations
        self.batch_size = batch_size
        # Model prameter
        self.weights_ = None
        self.bias_ = None

    def fit(self, X, y):
        """
        Train the model using stochastic gradient descent.

        Args:
            X: numpy array of shape (n_samples, n_features)
            y: numpy array of shape (n_samples,)
        """
        # Add bias column to the input X
        x_expand = np.concatenate((np.ones((X.shape[0], 1)), np.array(X)), axis=1)
        num_batches = x_expand.shape[0] // self.batch_size

                
        # Initializing weights with zeros. Therefore now shape : number of features + 1 for bias
        w = np.zeros((x_expand.shape[1], 1))
        
        for _ in range(self.n_iters):
            # shuffling data at each iteration to improve convergence speed
            indices = np.random.permutation(len(x_expand))
            x_sh = x_expand[indices]
            y_sh = np.array(y)[indices]
            for batch_num in range(num_batches):
                start = batch_num * self.batch_size
                end = (batch_num + 1) * self.batch_size
                
                # x_expand_batch = x_expand[start:end]
                # y_batch = np.array(y[start:end]).reshape((-1,1))
                x_expand_batch = x_sh[start:end]
                y_batch = y_sh[start:end].reshape((-1, 1))
                
                y_hat = x_expand_batch @ w
                error = y_batch - y_hat
            
                # Calculate Gradient: (2/N) * X^T * error
                gradient = - (1. / self.batch_size) * 2. * (x_expand_batch.T @ error)
                
                # Updated weights and move to opposite direction of the gradient
                w -= self.lr * gradient          
                
        # Store results back into weights_ and bias_
        self.bias_ = w[0]
        self.weights_ = w[1:]

    def predict(self, X):
        """
        Predict target values for the given input.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples,)
        """
        # y = X*w + b
        return (X @ self.weights_ + self.bias_).flatten()