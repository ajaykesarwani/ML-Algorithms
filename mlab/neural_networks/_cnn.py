import numpy as np

def get_patches(arr, patch_shape, strides):
    """
    Extract sliding window patches from 4D array for convolution.
    
    Args:
        arr: Input array of shape (batch_size, height, width, channels)
        patch_shape: Tuple (patch_h, patch_w) 
        strides: Tuple (stride_h, stride_w)
    
    Returns:
        Patches array for convolution operation
    """
    res = np.lib.stride_tricks.sliding_window_view(arr, patch_shape, axis=(1, 2))
    res = np.moveaxis(res, 3, 5)
    return res[:, ::strides[0], ::strides[1], ...]

def pad_images(X, padding):
    """
    Pad images with zeros on all sides.
    
    Args:
        X: Input images of shape (batch_size, height, width, channels)
        padding: Number of pixels to pad on each side
    
    Returns:
        Padded images
    """
    if padding == 0:
        return X
    return np.pad(X, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant')

class ConvLayer:
    """
    Convolutional layer for feature extraction with learnable filters.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1), padding='same', rng=None):
        self.in_channels = in_channels # int: Number of input channels
        self.out_channels = out_channels # int: Number of output channels (filters)
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size) # tuple: (height, width) of convolution kernel
        self.stride = stride if isinstance(stride, tuple) else (stride, stride) # tuple: (stride_h, stride_w) for convolution
        self.padding = padding # str: 'same' or 'valid' padding mode
        if rng is None:
            rng = np.random.RandomState()
            
        limit = np.sqrt(6 / (in_channels * self.kernel_size[0] * self.kernel_size[1] + out_channels))
        # kernel weights of shape (kH, kW, in_channels, out_channels)
        self.weight_ = rng.uniform(-limit, limit, (self.kernel_size[0], self.kernel_size[1], in_channels, out_channels)) 
        # bias of shape (out_channels,) 
        self.bias_ = np.zeros(out_channels) 
        
    def _pad(self, X, padding=None):
        if padding is None:
            padding = self.padding
        if padding == 'same':
            pad_x = (self.kernel_size[0]-1) //2 ,int(np.ceil((self.kernel_size[0]-1) /2))
            pad_y = (self.kernel_size[1]-1) //2 ,int(np.ceil((self.kernel_size[1]-1) /2))
            return np.pad(X, [(0,0), pad_x, pad_y, (0,0)])
        else:
            # Minimal architecture valid padding
            pad_h = max(0, self.kernel_size[0] - X.shape[1])
            pad_w = max(0, self.kernel_size[1] - X.shape[2])
            if pad_h > 0 or pad_w > 0:
                return np.pad(X, ((0,0), (0, pad_h), (0, pad_w), (0,0)), mode='constant')
        return X

    def get_output_shape(self, input_shape):
        h, w = input_shape[0], input_shape[1]
        if self.padding == 'same':
            out_h = int(np.ceil(h / self.stride[0]))
            out_w = int(np.ceil(w / self.stride[1]))
        else:
            eff_h = max(h, self.kernel_size[0])
            eff_w = max(w, self.kernel_size[1])
            out_h = (eff_h - self.kernel_size[0]) // self.stride[0] + 1
            out_w = (eff_w - self.kernel_size[1]) // self.stride[1] + 1
        return (out_h, out_w, self.out_channels)

    def __call__(self, X):
        self._input_shape = X.shape
        self._X_padded = self._pad(X)
        self._patches = get_patches(self._X_padded, self.kernel_size, self.stride)
        return np.einsum('bxyhwc,hwco->bxyo', self._patches, self.weight_) + self.bias_

    def backward(self, upstream_grad, alpha=None):
        # dW and dB (Reusing cached self._patches array)
        self._weight_grad = np.einsum('bxyhwc,bxyo->hwco', self._patches, upstream_grad) # ndarray: Gradient of the weight matrix
        if alpha is not None:
            # In-place L2 regularization gradient
            self._weight_grad += self.weight_ * alpha
        
        self._bias_grad = np.sum(upstream_grad, axis=(0, 1, 2)) # ndarray: Gradient of the bias vector
        
        # dX
        b, h_out, w_out, out_c = upstream_grad.shape # int: Number of samples, int: Output height, int: Output width, int: Number of output channels
        
        pad_h = self.kernel_size[0] - 1
        pad_w = self.kernel_size[1] - 1
        
        # Optimize memory: allocate grad_padded directly without intermediate dilated_grad
        H_padded = (h_out-1)*self.stride[0] + 1 + 2*pad_h
        W_padded = (w_out-1)*self.stride[1] + 1 + 2*pad_w
        grad_padded = np.zeros((b, H_padded, W_padded, out_c))
        grad_padded[:, pad_h:H_padded-pad_h:self.stride[0], pad_w:W_padded-pad_w:self.stride[1], :] = upstream_grad
        
        upstream_patches = get_patches(grad_padded, self.kernel_size, (1, 1))
        weight_flipped = np.flip(self.weight_, axis=(0, 1))
        
        dX_padded = np.einsum('hwco,bxyhwo->bxyc', weight_flipped, upstream_patches)
        
        # Remove padding from dX to match exactly self._input_shape
        _, h_in, w_in, _ = self._input_shape
        if self.padding == 'same':
            pad_h_top = (self.kernel_size[0] - 1) // 2 
            pad_w_left = (self.kernel_size[1] - 1) // 2 
            dX = dX_padded[:, pad_h_top:pad_h_top+h_in, pad_w_left:pad_w_left+w_in, :] 
        else:
            dX = dX_padded[:, 0:h_in, 0:w_in, :] 
            
        return dX

    def update(self, lr):
        # In-place parameter updates to avoid intermediate arrays
        self._weight_grad *= lr
        self.weight_ -= self._weight_grad
        self._bias_grad *= lr
        self.bias_ -= self._bias_grad


class PoolingLayer:
    """
    Max pooling layer for spatial dimension reduction.
    """
    def __init__(self, kernel_size, stride=(2,2), padding=0):
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size) # tuple: (height, width) of pooling window
        self.stride = stride if isinstance(stride, tuple) else (stride, stride) # tuple: (stride_h, stride_w) for pooling
        self.padding = padding # int: Padding amount

    def get_output_shape(self, input_shape):
        h, w = input_shape[0], input_shape[1]
        eff_h = max(h + 2 * self.padding, self.kernel_size[0])
        eff_w = max(w + 2 * self.padding, self.kernel_size[1])
        out_h = (eff_h - self.kernel_size[0]) // self.stride[0] + 1
        out_w = (eff_w - self.kernel_size[1]) // self.stride[1] + 1
        return (out_h, out_w, input_shape[2])

    def __call__(self, X):
        self._input_shape = X.shape # ndarray: Input for gradient computation in backward pass
        X_padded = pad_images(X, self.padding)
        # Minimal architecture valid padding
        pad_h = max(0, self.kernel_size[0] - X_padded.shape[1])
        pad_w = max(0, self.kernel_size[1] - X_padded.shape[2])
        if pad_h > 0 or pad_w > 0:
            X_padded = np.pad(X_padded, ((0,0), (0, pad_h), (0, pad_w), (0,0)), mode='constant')
        self._X_padded_shape = X_padded.shape # ndarray: Padded input shape
        patches = get_patches(X_padded, self.kernel_size, self.stride)
        
        b, out_h, out_w, kh, kw, c = patches.shape
        patches_flat = patches.reshape((b, out_h, out_w, kh * kw, c))
        self._max_idx = np.argmax(patches_flat, axis=3)
        
        return np.max(patches_flat, axis=3) # ndarray: Max pooled output

    def backward(self, upstream_grad):
        b, out_h, out_w, c = upstream_grad.shape
        dX_padded = np.zeros(self._X_padded_shape)
        
        # Vectorized gradient routing
        n_idx = np.arange(b)[:, None, None, None]
        i_idx = np.arange(out_h)[None, :, None, None]
        j_idx = np.arange(out_w)[None, None, :, None]
        c_idx = np.arange(c)[None, None, None, :]
        
        kh, kw = self.kernel_size
        stride_h, stride_w = self.stride
        
        max_h = (i_idx * stride_h) + (self._max_idx // kw)
        max_w = (j_idx * stride_w) + (self._max_idx % kw)
        
        np.add.at(dX_padded, (n_idx, max_h, max_w, c_idx), upstream_grad)
                        
        _, h_in, w_in, _ = self._input_shape
        return dX_padded[:, self.padding:self.padding+h_in, self.padding:self.padding+w_in, :]


class ModularLinearLayer:
    """
    Modular linear layer connecting flattened conv features to output classes.
    """
    def __init__(self, in_features, out_features, rng=None):
        self.in_features = in_features # int: Number of input features
        self.out_features = out_features # int: Number of output features
        if rng is None:
            rng = np.random.RandomState()
            
        limit = np.sqrt(6 / (in_features + out_features))
        self.weight_ = rng.uniform(-limit, limit, (in_features, out_features)) # ndarray: Weight matrix of shape (in_features, out_features)
        self.bias_ = np.zeros(out_features) # ndarray: Bias vector of shape (out_features,)

    def __call__(self, X):
        self._prev_input = X  # ndarray: Input to the layer, used for backpropagation
        return np.dot(X, self.weight_) + self.bias_

    def backward(self, upstream_grad, alpha=None):
        self._weight_grad = np.dot(self._prev_input.T, upstream_grad) # ndarray: Gradient of the weight matrix
        if alpha is not None:
            # In-place L2 regularization gradient
            self._weight_grad += self.weight_ * alpha
            
        self._bias_grad = np.sum(upstream_grad, axis=0) # ndarray: Gradient of the bias vector
        return np.dot(upstream_grad, self.weight_.T)

    def update(self, lr):
        # In-place parameter updates to avoid intermediate arrays
        self._weight_grad *= lr
        self.weight_ -= self._weight_grad
        self._bias_grad *= lr
        self.bias_ -= self._bias_grad

    def __repr__(self):
        return f"ModularLinearLayer(in_features={self.in_features}, out_features={self.out_features})"


class ReLULayer:
    """
    ReLU activation layer: f(x) = max(0, x)
    """
    def __init__(self):
        pass

    def __call__(self, X):
        self._prev_result = np.maximum(X, 0, out=X) # ndarray: Previous activation output for derivative calculation
        return self._prev_result

    def backward(self, upstream_grad):
        upstream_grad *= (self._prev_result > 0)
        return upstream_grad


class SoftmaxLayer:
    """
    Softmax activation layer for multi-class probability output.
    """
    def __init__(self):
        pass

    def __call__(self, X):
        shifted_X = X - np.max(X, axis=1, keepdims=True)
        np.exp(shifted_X, out=shifted_X)
        # In-place division to avoid intermediate array
        shifted_X /= np.sum(shifted_X, axis=1, keepdims=True)
        self._prev_result = shifted_X  # ndarray: Previous softmax output for derivative calculation
        return self._prev_result

    def backward(self, upstream_grad):
        # vectorized multiplication
        dX = self._prev_result * upstream_grad
        dX -= self._prev_result * np.sum(dX, axis=1, keepdims=True)
        return dX


class CNNClassifier:
    """
    Convolutional Neural Network classifier for image classification.
    """
    def __init__(self, layers, lr=0.01, epochs=50, random_state=None, alpha=0.0001, batch_size=32, input_shape=(28,28,1)):
        self.layers = layers # list: Sequence of ConvLayer/PoolingLayer objects
        self.layers_ = self.layers # list: Copy of layers for modification during training
        self.lr = lr # float: Learning rate
        self.epochs = epochs # int: Number of training epochs
        self.batch_size = batch_size # int: Mini-batch size for training
        self.random_state = random_state # int: Random seed for reproducibility
        self.alpha = alpha  # float: L2 regularization parameter
        self.input_shape = input_shape # tuple: (height, width, channels) of input images

    def fit(self, X, y):
        self.rng_ = np.random.RandomState(self.random_state) # RandomState: Random number generator
        
        curr_shape = self.input_shape
        for layer in self.layers:
            if hasattr(layer, 'get_output_shape'):
                curr_shape = layer.get_output_shape(curr_shape)
        
        in_features = int(np.prod(curr_shape))
        
        # If there are no classes defined explicitly (like y values > num_classes)
        num_classes = len(np.unique(y))
        self.out_layer_ = ModularLinearLayer(in_features, num_classes, rng=self.rng_) # ModularLinearLayer: Final classification layer
        self.softmax_layer_ = SoftmaxLayer() # SoftmaxLayer: Final activation layer
        
        n_samples = X.shape[0]
        
        for epoch in range(self.epochs):
            indices = np.arange(n_samples)
            self.rng_.shuffle(indices)
            
            for start_idx in range(0, n_samples, self.batch_size):
                batch_idx = indices[start_idx:start_idx + self.batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                
                # Forward
                out = self._forward(X_batch)
                
                # Cross entropy grad: Avoid creating intermediate y_one_hot array
                grad_ce = np.zeros_like(out)
                grad_ce[np.arange(len(y_batch)), y_batch] = -1.0 / (out[np.arange(len(y_batch)), y_batch] + 1e-9) / len(y_batch)
                
                # Backward
                grad = self.softmax_layer_.backward(grad_ce)
                grad = self.out_layer_.backward(grad, alpha=self.alpha)
                
                grad = grad.reshape(-1, *curr_shape)
                
                for layer in reversed(self.layers):
                    if isinstance(layer, ConvLayer) or isinstance(layer, ModularLinearLayer):
                        if 'alpha' in layer.backward.__code__.co_varnames:
                            grad = layer.backward(grad, alpha=self.alpha)
                        else:
                            grad = layer.backward(grad)
                        layer.update(self.lr)
                    elif hasattr(layer, 'update'):
                        grad = layer.backward(grad)
                        layer.update(self.lr)
                    else:
                        grad = layer.backward(grad)
                        
            # Update out_layer_
            self.out_layer_.update(self.lr)
            
        return self

    def predict(self, X):
        probs = self._forward(X)
        return np.argmax(probs, axis=1)
        
    def _forward(self, X):
        out = X
        for layer in self.layers:
            out = layer(out)
        out = out.reshape(out.shape[0], -1)
        out = self.out_layer_(out)
        out = self.softmax_layer_(out)
        return out

    def _calculate_loss(self, logits, y):
        # Avoid creating intermediate y_one_hot array
        correct_probs = logits[np.arange(len(y)), y]
        eps = 1e-9
        ce_loss = -np.mean(np.log(correct_probs + eps))
        
        l2_loss = 0
        for layer in self.layers:
            if hasattr(layer, 'weight_'):
                # Avoid temporary matrix allocation with vdot instead of ** 2
                l2_loss += np.vdot(layer.weight_, layer.weight_)
        l2_loss += np.vdot(self.out_layer_.weight_, self.out_layer_.weight_)
        
        return ce_loss, 0.5 * self.alpha * l2_loss

    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)
