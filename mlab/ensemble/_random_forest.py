import numpy as np
from mlab.trees._tree_utils import Node, gini_impurity, majority_class, compute_class_weights


class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2, min_samples_leaf=1, class_weight=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight  # dict or None
        self.feature_importances_ = None
        self.feature_names_ = None
        self.root = None
        self._actual_depth = 0

    def fit(self, X, y, feature_names=None):
        """
        Build the decision tree from training data.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        # If no class_weight provided, compute from data (to handle imbalance)
        if self.class_weight is None:
            self.class_weight = compute_class_weights(y)

        n_samples, n_features = X.shape
        self.feature_importances_ = np.zeros(n_features, dtype=np.float64)
        self.feature_names_ = feature_names
        self._actual_depth = 0

        self.root = self._grow_tree(X, y, depth=0)

        total_importance = float(np.sum(self.feature_importances_))
        if total_importance > 0.0:
            self.feature_importances_ /= total_importance

        return self

    def _best_split(self, X, y, n_samples, n_features):
        """Find the optimal feature index and threshold."""
        best_gain = -1.0
        split_idx, split_thresh = None, None
        current_gini = gini_impurity(y)

        for feat_idx in range(n_features):
            X_column = X[:, feat_idx]
            sort_idx = np.argsort(X_column)
            X_sort = X_column[sort_idx]
            y_sort = y[sort_idx]

            max_label = int(np.max(y_sort))
            left_counts = np.zeros(max_label + 1, dtype=np.int64)
            right_counts = np.bincount(y_sort, minlength=max_label + 1)

            n_left = 0
            n_right = n_samples

            for i in range(n_samples - 1):
                label = y_sort[i]
                left_counts[label] += 1
                right_counts[label] -= 1
                n_left += 1
                n_right -= 1

                # enforce min_samples_leaf on both sides
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                # skip identical feature values
                if X_sort[i] == X_sort[i + 1]:
                    continue

                p_left = left_counts / n_left
                p_right = right_counts / n_right
                gini_left = 1.0 - np.sum(p_left ** 2)
                gini_right = 1.0 - np.sum(p_right ** 2)

                gain = current_gini - (
                    (n_left / n_samples) * gini_left
                    + (n_right / n_samples) * gini_right
                )

                # class-weight boost for the minority class
                if self.class_weight is not None and len(self.class_weight) > 0:
                    minority_label = max(self.class_weight, key=self.class_weight.get)
                    n_min_left = np.sum(y_sort[: i + 1] == minority_label)
                    n_min_right = np.sum(y_sort[i + 1 :] == minority_label)
                    weight_factor = (
                        n_min_left / (n_left + 1e-8)
                        + n_min_right / (n_right + 1e-8)
                    ) * 0.5  # mild boost
                    gain *= (1.0 + weight_factor)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = (X_sort[i] + X_sort[i + 1]) / 2.0

        return split_idx, split_thresh, best_gain

    def _grow_tree(self, X, y, depth):
        self._actual_depth = max(self._actual_depth, depth)
        n_samples, n_features = X.shape

        # guard: no samples -> default leaf
        if n_samples == 0:
            return Node(value=0)

        num_labels = len(np.unique(y))

        # stopping conditions (early stopping)
        if (
            depth >= self.max_depth
            or num_labels == 1
            or n_samples < self.min_samples_split
            or n_samples < 2 * self.min_samples_leaf
        ):
            leaf_value = self._weighted_majority(y)
            return Node(value=leaf_value)

        feat_idx, thresh, gain = self._best_split(X, y, n_samples, n_features)

        if gain <= 0.0 or feat_idx is None or thresh is None:
            leaf_value = self._weighted_majority(y)
            return Node(value=leaf_value)

        # accumulate feature importance (normalized later)
        self.feature_importances_[feat_idx] += (n_samples / len(y)) * gain

        left_mask = X[:, feat_idx] <= thresh
        right_mask = ~left_mask

        left_child = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feature=feat_idx, threshold=thresh, left=left_child, right=right_child)

    def _weighted_majority(self, y):
        """
        Choose leaf class as weighted majority according to class_weight.
        """
        y = np.asarray(y, dtype=np.int64)
        if y.size == 0:
            return 0

        counts = np.bincount(y)
        weights = np.ones_like(counts, dtype=float)

        if self.class_weight is not None:
            for label, w in self.class_weight.items():
                if label < len(weights):
                    weights[label] = w

        weighted_counts = counts * weights
        return int(np.argmax(weighted_counts))

    def predict(self, X):
        """
        Predict class labels for the given input array.
        """
        if self.root is None:
            raise ValueError("DecisionTree has not been fitted yet")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")

        return np.array(
            [self._traverse_tree(x, self.root) for x in X],
            dtype=np.int64,
        )

    def _traverse_tree(self, x, node):
        if node.is_leaf:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    @property
    def tree_depth(self):
        """Return the actual maximum depth reached by the tree."""
        return self._actual_depth

    def get_depth(self):
        return self._actual_depth