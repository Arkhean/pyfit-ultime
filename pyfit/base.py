"""
Core definitions for models and training
"""

# pylint: disable=unused-import

# Temporarily disable annoying Pylint check
# pylint: disable=too-few-public-methods

import numpy as np

# A np.ndarray is a multidimensional array
# Use NumPy n-dimensional array class as our np.ndarray class
from numpy import ndarray


class BaseEstimator:
    """
    Abstract base class for all estimators
    """

    x_train: np.ndarray = None
    y_train: np.ndarray = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fit the model using x_train as training data and y_train as target values
        """
        # Stores training data and targets for use in derived classes
        self.x_train = x_train
        self.y_train = y_train
