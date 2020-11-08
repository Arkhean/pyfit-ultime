"""
module containing metrics for model evaluation
"""

#rappel pour moi:
# TP: pred=True, true=True
# FP: pred = True, true= False
# FN: pred = False, true = True
# TN: pred= False, true = False
import numpy as np

def accuracy_score(y_true, y_pred):
    """
    Accuracy classification score.
    """
    exact=0
    total=len(y_true)
    for i,y_true_i in enumerate(y_true):
        if y_true_i==y_pred[i]:
            exact+=1
    return(exact/total)

def precision_score(y_true, y_pred):
    """
    The precision is the ratio tp / (tp + fp) where tp is the number of true
    positives and fp the number of false positives. The precision is intuitively
     the ability of the classifier not to label as positive a sample that is
     negative.
    """
    TP=0
    FP=0
    for i,y_pred_i in enumerate(y_pred):
        if y_pred_i:
            if y_pred_i==y_true[i]:
                TP+=1
            else:
                FP+=1
    if TP==0:
        return 0
    else:
        return(TP/(TP+FP))


def recall_score(y_true, y_pred):
    """
    The recall is the ratio tp / (tp + fn) where tp is the number of true
    positives and fn the number of false negatives. The recall is intuitively
    the ability of the classifier to find all the positive samples.
    """
    TP=0
    FN=0
    for i,y_true_i in enumerate(y_true):
        if y_true_i:
            if y_true_i==y_pred[i]:
                TP+=1
            else:
                FN+=1
    if TP==0:
        return 0
    else:
        return(TP/(TP+FN))

