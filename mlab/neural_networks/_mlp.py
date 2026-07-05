import numpy as np

class ModularLinearLayer:
    """Fully connected (dense) layer."""

    def __init__(self, input_size, output_size, rng=None):
        rng = rng if rng is not None else np.random.default_rng()

        limit = np.sqrt(1.0 / input_size)
        # weight matrix of shape (input_size, output_size)
        self.weight = rng.uniform(-limit, limit, size=(input_size, output_size)) 
        # bias vector of shape (output_size,)
        self.bias = np.zeros(output_size) 

        # Populated during forward/backward passes.
        self.X = None
        self.grad_weight = None 
        self.grad_bias = None

    def __call__(self, X):
        """Forward pass: ``X @ weight + bias``."""
        self.X = X
        return X @ self.weight + self.bias

    def backward(self, grad_output):
        """Backward pass: compute gradients w.r.t. input, weight, and bias."""

        self.grad_weight = self.X.T @ grad_output 
        self.grad_bias = np.sum(grad_output, axis=0)
        
        grad_input = grad_output @ self.weight.T
        return grad_input

    def update(self, learning_rate, alpha=0.0):
        """Update weights and biases using computed gradients."""
        self.weight -= learning_rate * (self.grad_weight + alpha * self.weight)
        self.bias -= learning_rate * self.grad_bias


class SigmoidLayer:
    """Sigmoid activation function."""
    def __init__(self):
        self.X = None
        self.output = None

    def __call__(self, X):
        """Forward: 1 / (1 + exp(-X))"""
        # Clip to avoid overflow warnings for very large X
        self.output = 1.0 / (1.0 + np.exp(-np.clip(X, -500, 500)))
        return self.output

    def backward(self, grad_output):
        """Backward: grad * sigmoid(X) * (1 - sigmoid(X))"""
        return grad_output * self.output * (1.0 - self.output)


class TanhLayer:
    """Tanh activation function."""

    def __init__(self):
        self.X = None
        self.output = None

    def __call__(self, X):
        self.output = np.tanh(X)
        return self.output

    def backward(self, grad_output):
        return grad_output * (1.0 - self.output ** 2)


class ReLULayer:
    """ReLU activation function."""

    def __init__(self):
        self.X = None
        self.output = None

    def __call__(self, X):
        self.X = X
        self.output = np.maximum(0.0, X)
        return self.output

    def backward(self, grad_output):
        return grad_output * (self.X > 0).astype(grad_output.dtype)


class SoftmaxLayer:
    """Softmax activation (for multi-class output)."""

    def __init__(self):
        self.X = None
        self.output = None

    def __call__(self, X):
        shifted = X - np.max(X, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        self.output = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.output

    def backward(self, grad_output):
        n_samples, _ = self.output.shape
        grad_input = np.empty_like(grad_output)
        for i in range(n_samples):
            y = self.output[i].reshape(-1, 1)               # (n_classes, 1)
            jacobian = np.diagflat(y) - y @ y.T              # (n_classes, n_classes)
            grad_input[i] = jacobian @ grad_output[i]
        return grad_input

_ACTIVATIONS = {
    "relu": ReLULayer,
    "sigmoid": SigmoidLayer,
    "tanh": TanhLayer,
}

# --------------------------------------------------------------------------- #

class MLPRegressor:
    def __init__(self, hidden_layer_sizes=(50, 30), lr=0.01, epochs=100,
                 random_state=None, alpha=0.0001, activation="relu",
                 learning_rate=None, n_iterations=None, max_iter=None):

        self.hidden_layer_sizes = hidden_layer_sizes

        self.lr = learning_rate if learning_rate is not None else lr
        self.learning_rate = self.lr  # alias kept in sync

        if n_iterations is not None:
            self.epochs = n_iterations
        elif max_iter is not None:
            self.epochs = max_iter
        else:
            self.epochs = epochs
        self.n_iterations = self.epochs
        self.max_iter = self.epochs

        self.random_state = random_state
        self.alpha = alpha # L2 regularization strength

        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'. "
                f"Expected one of {list(_ACTIVATIONS)}."
            )
        self.activation = activation

        self.layers_ = []       # list of layer objects (set after fit)
        self.loss_curve_ = []   
        self._y_was_1d = False

    def _build_network(self, n_features, n_outputs, rng):
        """Construct the list of layers: Linear -> Activation -> ... -> Linear."""
        activation_cls = _ACTIVATIONS[self.activation]
        layer_sizes = [n_features] + list(self.hidden_layer_sizes) + [n_outputs]

        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(ModularLinearLayer(layer_sizes[i], layer_sizes[i + 1], rng=rng))
            is_output_layer = (i == len(layer_sizes) - 2)
            if not is_output_layer:
                layers.append(activation_cls())
        return layers

    def _forward(self, X):
        """Run X through every layer in order, returning the final output."""
        out = X
        for layer in self.layers_:
            out = layer(out)
        return out

    def _backward_and_update(self, grad):
        for layer in reversed(self.layers_):
            grad = layer.backward(grad)
        for layer in self.layers_:
            if hasattr(layer, "update"):
                layer.update(self.lr, self.alpha)

    def fit(self, X, y):
        """
        Train the network using backpropagation.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        self._y_was_1d = (y.ndim == 1)
        if self._y_was_1d:
            y = y.reshape(-1, 1)

        n_samples, n_features = X.shape
        n_outputs = y.shape[1]

        rng = np.random.default_rng(self.random_state)
        self.layers_ = self._build_network(n_features, n_outputs, rng)
        self.loss_curve_ = []

        for _ in range(self.epochs):
            # ----- forward pass -----
            y_pred = self._forward(X)

            # ----- loss (MSE + L2 penalty, for monitoring only) -----
            mse = np.mean((y_pred - y) ** 2)
            l2_penalty = self.alpha * sum(
                np.sum(layer.weight ** 2)
                for layer in self.layers_
                if hasattr(layer, "weight")
            )
            self.loss_curve_.append(mse + l2_penalty)

            # ----- backward pass -----
            # d(MSE)/d(y_pred) = 2 * (y_pred - y) / n_samples
            grad = 2.0 * (y_pred - y) / n_samples
            self._backward_and_update(grad)

        return self

    def predict(self, X):
        """
        Predict target values.
        """
        if not self.layers_:
            raise RuntimeError("This MLPRegressor instance is not fitted yet. "
                                "Call 'fit' before using 'predict'.")

        X = np.asarray(X, dtype=float)
        y_pred = self._forward(X)

        if getattr(self, "_y_was_1d", True) and y_pred.shape[1] == 1:
            return y_pred.ravel()
        return y_pred
