"""
data processing functions
"""

import numpy as np


def train_test_split(*arrays, **options):
    """
    split data between train and test set with ratio
    """
    if len(arrays) == 0:
        raise ValueError("At least one array required as input")
    test_size = options.get('test_size', 0.2) # default is 0.2

    indices = np.random.permutation(len(arrays[0]))
    split_index = int((1 - test_size) * len(arrays[0]))
    training_idx, test_idx = indices[:split_index], indices[split_index:]
    res = ()
    for array in arrays:
        train_set, test_set = array[training_idx], array[test_idx]
        res += (train_set, test_set)
    return res

def one_hot_encode(x):
    """
    transform categorical data to vector with one one and zeros
    """
    categories = dict()
    index = 0
    for element in x:
        if x not in categories:
            categories[x] = index
            index += 1
    n = len(categorie)
    res = np.zeros((len(x), n))
    for i in range(len(x)):
        res[i, categories[x]] = 1
    return res


class Scaler:
    def __init__(self):
        pass

    def fit(self):
        pass

    def transform(self):
        pass
