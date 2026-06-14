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
    p = counts / m
    return 1.0 - np.sum(p ** 2)


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