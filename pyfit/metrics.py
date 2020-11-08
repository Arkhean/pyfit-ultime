"""
module containing metrics for model evaluation
"""

import numpy as np

def accuracy_score(y_true, y_pred):
    """
    Accuracy classification score.
    """

def precision_score(y_true, y_pred):
    """
    The precision is the ratio tp / (tp + fp) where tp is the number of true
    positives and fp the number of false positives. The precision is intuitively
     the ability of the classifier not to label as positive a sample that is
     negative.
    """

def recall_score(y_true, y_pred):
    """
    The recall is the ratio tp / (tp + fn) where tp is the number of true
    positives and fn the number of false negatives. The recall is intuitively
    the ability of the classifier to find all the positive samples.
    """
