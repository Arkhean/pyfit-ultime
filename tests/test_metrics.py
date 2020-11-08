from pyfit.metrics import *

def test_accuracy_score():
    y_true=[True,False,False,True]
    y_pred=[False,True,True,False]
    assert accuracy_score(y_true, y_pred) == 0

    y_true = [True,True,True,False]
    y_pred=[False,True,True,False]
    assert accuracy_score(y_true,y_pred) == 0.75

    y_true = [True,True,True,False]
    y_pred=[False,False,True,False]
    assert accuracy_score(y_true,y_pred) == 0.5

def test_precision_score():
    y_true=[True,False,False,True] # TP = 0
    y_pred=[False,True,True,False] # FP = 2
    assert precision_score(y_true, y_pred) == 0

    y_true = [True,True,True,False]  # TP = 2
    y_pred = [False,True,True,False] # FP = 0
    assert precision_score(y_true,y_pred) == 1

    y_true = [True,True,True,False]   # TP = 1
    y_pred = [False,False,True,True] # FP = 1
    assert precision_score(y_true,y_pred) == 0.5

def test_recall_score():
    y_true=[True,False,False,True] # TP = 0
    y_pred=[False,True,True,False] # FN = 2
    assert recall_score(y_true, y_pred) == 0

    y_true = [True,True,True,False]  # TP = 2
    y_pred = [False,True,True,False] # FN = 1
    assert recall_score(y_true,y_pred) == 2/3

    y_true = [True,True,True,False]   # TP = 1
    y_pred = [False,False,True,False] # FN = 2
    assert recall_score(y_true,y_pred) == 1/3

