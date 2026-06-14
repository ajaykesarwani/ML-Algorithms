import numpy as np
from mlab.trees._decision_tree import DecisionTree
from mlab.trees._tree_utils import set_global_seed


class RandomForest:
    def __init__(
        self,
        n_estimators=20,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        n_features_to_consider="sqrt",
        random_state=42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_features_to_consider = n_features_to_consider
        self.random_state = random_state

        self.trees_ = []
        self.feature_importances_ = None
        self.feature_names_ = None

        set_global_seed(random_state)

    def fit(self, X, y, feature_names=None):
        """
        Build the random forest from training data using bootstrap sampling.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        n_samples, n_features = X.shape
        self.trees_ = []
        self.feature_names_ = feature_names

        aggregated_importances = np.zeros(n_features, dtype=np.float64)

        for _ in range(self.n_estimators):
            # row bootstrap
            bootstrap_idx = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[bootstrap_idx]
            y_boot = y[bootstrap_idx]

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(X_boot, y_boot, feature_names=feature_names)

            self.trees_.append(tree)
            if tree.feature_importances_ is not None:
                aggregated_importances += tree.feature_importances_

        self.feature_importances_ = aggregated_importances / self.n_estimators
        return self

    def predict(self, X):
        """
        Predict class labels using majority voting across all trees.
        """
        if not self.trees_:
            raise ValueError("RandomForest has not been fitted yet")

        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]

        tree_preds = np.array([tree.predict(X) for tree in self.trees_])

        final_preds = np.zeros(n_samples, dtype=np.int64)
        for i in range(n_samples):
            votes = tree_preds[:, i]
            final_preds[i] = np.bincount(votes).argmax()

        return final_preds