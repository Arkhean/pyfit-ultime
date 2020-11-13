"""
loss functions
"""

import math
import numpy as np

from _engine import as_array

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error regression loss"""
    y_true = as_array(y_true)
    y_pred = as_array(y_pred)
    return np.mean((y_true - y_pred) ** 2)



def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error regression loss"""
    y_true = as_array(y_true)
    y_pred = as_array(y_pred)
    return np.mean(np.abs(y_true - y_pred))



def log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Log loss, aka logistic loss or cross-entropy loss."""
    y_true = as_array(y_true)
    y_pred = as_array(y_pred)
    n = len(y_pred)
    s_ = 0
    for i, y_i in enumerate(y_true):
        s_ += y_i*math.log(y_pred[i]) + (1-y_i)*math.log(1-y_pred[i])
    logl = (-1 / n) * s_
    return logl
