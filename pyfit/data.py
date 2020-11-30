"""
data processing functions
"""
from typing import Any, List
import numpy as np


def train_test_split(*arrays: Any, **options: Any) -> List[Any]:
    """
    split data between train and test set with ratio
    """
    if len(arrays) == 0:
        raise ValueError("At least one array required as input")
    test_size = options.get('test_size', 0.2) # default is 0.2

    indices = np.random.permutation(len(arrays[0]))
    split_index = int((1 - test_size) * len(arrays[0]))
    training_idx, test_idx = indices[:split_index], indices[split_index:]
    res = list()
    for array in arrays:
        train_set, test_set = array[training_idx], array[test_idx]
        res += [train_set, test_set]
    return res

def one_hot_encode(x: np.ndarray) -> np.ndarray:
    """
    transform categorical data to vector with one one and zeros
    """
    categories = dict()
    index = 0
    for element in x:
        if element not in categories:
            categories[x] = index
            index += 1
    cat_count = len(categories)
    res = np.zeros((len(x), cat_count))
    for i in range(len(x)):
        res[i, categories[x]] = 1
    return res

def make_classification(n_samples: int, nb_class: int) -> np.ndarray:
    """
    generate clusters of points normally distributed
    """
    # On pose un centre par classe...
    pass

class Scaler:
    """
    Standardize features by removing the mean and scaling to unit variance
    """
    def __init__(self) -> None:
        pass

    def fit(self, x: np.ndarray) -> None:
        """
        Compute the mean and std to be used for later scaling.
        """

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Perform standardization by centering and scaling
        """
