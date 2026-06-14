import numpy as np


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    @property
    def is_leaf(self):
        return self.value is not None


def gini_impurity(y):
    """
    Compute Gini impurity
    """
    y = np.asarray(y, dtype=np.int64)
    m = len(y)
    if m == 0:
        return 0.0
    counts = np.bincount(y)
    probabilities = counts / m
    return 1.0 - np.sum(probabilities ** 2)


def majority_class(y):
    """
    Return the majority class label in y.
    """
    y = np.asarray(y, dtype=np.int64)
    if y.size == 0:
        return 0
    return int(np.bincount(y).argmax())


def set_global_seed(random_state=None):
    """
    Set NumPy's global random seed for reproducibility.
    """
    if random_state is not None:
        np.random.seed(random_state)


def compute_class_weights(y):
    """
    Compute per-class weights for imbalanced data
    """
    y = np.asarray(y, dtype=np.int64)
    counts = np.bincount(y)
    total = counts.sum()
    n_classes = len(counts)
    # avoid division by zero if some class is missing
    weights = {}
    for label in range(n_classes):
        if counts[label] == 0:
            weights[label] = 0.0
        else:
            weights[label] = float(total / (n_classes * counts[label]))
    return weights