"""
this module include scaling, one hot encoding
"""
from typing import Any, List, Dict
import numpy as np
from pyfit.engine import as_array

class Scaler:
    """
    Standardize features by removing the mean and scaling to unit variance
    """
    def __init__(self) -> None:
        self.mean_ = None
        self.std_ = None

    def fit(self, x: np.ndarray) -> None:
        """
        Compute the mean and std to be used for later scaling.
        """
        self.mean_ = np.mean(x, axis=0)
        self.std_ = np.std(x, axis=0)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Perform standardization by centering and scaling
        """
        x = as_array(x)
        if self.mean_ is not None and self.std_ is not None:
            return (x - self.mean_) / self.std_
        raise RuntimeError("method fit has not been called yet")

################################################################################
class OneHotEncoder():
    """
    transform categorical data to vector with one one and zeros
    """
    def __init__(self) -> None:
        """initialize self"""
        self.categories_: List[Dict[Any, int]] = list()
        self.n_features = 0
        self.cat_count: List[int] = list()

    def fit(self, x: np.ndarray) -> None:
        """Fit OneHotEncoder to X."""
        x = as_array(x)
        # reset categories
        self.categories_ = list()
        if len(x.shape) < 2:
            self.n_features = 1
            x = np.expand_dims(x, axis=1)
        else:
            self.n_features = x.shape[1]
        # find all categories and link to number
        for feature in range(self.n_features):
            self.categories_.append(dict())
            index = 0
            for element in x[:, feature]:
                if element not in self.categories_[feature]:
                    self.categories_[feature][element] = index
                    index += 1
        self.cat_count = [len(d) for d in self.categories_]

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform X using one-hot encoding."""
        x = as_array(x)
        if self.n_features < 2:
            x = np.expand_dims(x, axis=1)
        res = np.zeros((len(x), sum(self.cat_count)))
        for feature in range(self.n_features):
            delta = sum(self.cat_count[:feature])
            for i in range(len(x)):
                if x[i, feature] not in self.categories_[feature]:
                    raise RuntimeError(f'unknown categorical feature encountered: {x[i, feature]}')
                res[i, self.categories_[feature][x[i, feature]]+delta] = 1
        return res
