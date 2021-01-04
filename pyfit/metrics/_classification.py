"""
module containing metrics for model evaluation
"""

#rappel pour moi:
# TP: pred=True, true=True
# FP: pred = True, true= False
# FN: pred = False, true = True
# TN: pred= False, true = False
from typing import List
import numpy as np
from pyfit.engine import Tensor

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Accuracy classification score.
    """
    exact = 0
    total = len(y_true)
    for i, y_true_i in enumerate(y_true):
        if y_true_i == y_pred[i]:
            exact += 1
    return exact / total

def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    The precision is the ratio tp / (tp + fp) where tp is the number of true
    positives and fp the number of false positives. The precision is intuitively
     the ability of the classifier not to label as positive a sample that is
     negative.
    """
    true_pos = 0
    false_pos = 0
    for i, y_pred_i in enumerate(y_pred):
        if y_pred_i:
            if y_pred_i == y_true[i]:
                true_pos += 1
            else:
                false_pos += 1
    if true_pos == 0:
        return 0
    return true_pos / (true_pos + false_pos)


def recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    The recall is the ratio tp / (tp + fn) where tp is the number of true
    positives and fn the number of false negatives. The recall is intuitively
    the ability of the classifier to find all the positive samples.
    """
    true_pos = 0
    false_neg = 0
    for i, y_true_i in enumerate(y_true):
        if y_true_i:
            if y_true_i == y_pred[i]:
                true_pos += 1
            else:
                false_neg += 1
    if true_pos == 0:
        return 0
    return true_pos / (true_pos + false_neg)

################################################################################

def binary_accuracy(y_true: List[Tensor], y_pred: List[Tensor]) -> float:
    """Binary accuracy, used by Trainer"""

    n_exact: int = sum(
        [
            np.array_equal(y_true_i.data, np.round(y_pred_i.data))
            for (y_true_i, y_pred_i) in zip(y_true, y_pred)
        ]
    )
    n_total: int = max(len(y_true), 1)
    return n_exact / n_total
