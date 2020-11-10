import numpy as np
import math
# MSE

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if type(y_true) is not np.ndarray:
        y_true = np.array(y_true)
    if type(y_pred) is not np.ndarray:
        y_pred = np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)

# MAE

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if type(y_true) is not np.ndarray:
        y_true = np.array(y_true)
    if type(y_pred) is not np.ndarray:
        y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))


# log loss
def log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    n=len(y_pred)
    sum=0
    for i,yi in enumerate(y_true):
        sum+= yi*math.log(y_pred[i]) + (i-yi)*math.log(1-y_pred[i])
    logl=(-1/n)*sum
    return logl
