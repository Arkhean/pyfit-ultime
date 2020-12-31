"""
loss functions
"""

import math
from typing import Union
import numpy as np
from pyfit.engine import as_tensor, as_array
from pyfit.engine import Tensor


def mean_squared_error(
        y_true: Union[Tensor, np.ndarray],
        y_pred: Union[Tensor, np.ndarray]
    ) -> Union[Tensor, np.ndarray]:
    """Mean squared error regression loss"""
    if isinstance(y_true, Tensor) or isinstance(y_pred, Tensor):
        y_true = as_tensor(y_true)
        y_pred = as_tensor(y_pred)
        return ((y_true - y_pred) ** 2).mean()
    y_true = as_array(y_true)
    y_pred = as_array(y_pred)
    return np.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error regression loss"""
    if isinstance(y_true, Tensor) or isinstance(y_pred, Tensor):
        y_true = as_tensor(y_true)
        y_pred = as_tensor(y_pred)
        return ((y_true - y_pred).abs()).mean()
    y_true = as_array(y_true)
    y_pred = as_array(y_pred)
    return np.mean(np.abs(y_true - y_pred))

def log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Log loss, aka logistic loss or cross-entropy loss."""
    size = len(y_pred)
    sum_ = 0
    if isinstance(y_true, Tensor) or isinstance(y_pred, Tensor):
        y_true = as_tensor(y_true)
        y_pred = as_tensor(y_pred)
        for i, y_i in enumerate(y_true):
            sum_ += y_i * (y_pred[i]).log() + (1 - y_i) * (1 - y_pred[i]).log()
        logl = (-1 / size) * sum_
    else:
        y_true = as_array(y_true)
        y_pred = as_array(y_pred)
        for i, y_i in enumerate(y_true):
            sum_ += y_i * math.log(y_pred[i]) + (1 - y_i) * math.log(1 - y_pred[i])
        logl = (-1 / size) * sum_
    return logl
