"""
loss functions
"""

import math
import numpy as np
from pyfit.engine import as_tensor
from pyfit.engine import Tensor


def mean_squared_error(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """Mean squared error regression loss"""
    y_true = as_tensor(y_true)
    y_pred = as_tensor(y_pred)
    return ((y_true - y_pred) ** 2).mean()

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error regression loss"""
    y_true = as_tensor(y_true)
    y_pred = as_tensor(y_pred)
    return (np.abs(y_true - y_pred)).mean()

def log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Log loss, aka logistic loss or cross-entropy loss."""
    y_true = as_tensor(y_true)
    y_pred = as_tensor(y_pred)
    size = len(y_pred)
    sum_ = 0
    for i, y_i in enumerate(y_true):
        sum_ += y_i * math.log(y_pred[i]) + (1 - y_i) * math.log(1 - y_pred[i])
    logl = (-1 / size) * sum_
    return logl
