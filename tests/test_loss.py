from pyflit.loss import *

# tests savamment empreintés depuis les exemples données sur le site de
# scikit-learn

def test_mean_squarred_error():
    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]
    assert mean_squared_error(y_true, y_pred) == 0.375

    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]
    assert mean_squared_error(y_true, y_pred, squared=False) >= 0.612

    y_true = [[0.5, 1],[-1, 1],[7, -6]]
    y_pred = [[0, 2],[-1, 2],[8, -5]]
    assert mean_squared_error(y_true, y_pred) >= 0.708


def test_mean_absolute_error() :
    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]
    assert mean_absolute_error(y_true, y_pred) == 0.5

    y_true = [[0.5, 1], [-1, 1], [7, -6]]
    y_pred = [[0, 2], [-1, 2], [8, -5]]
    assert mean_absolute_error(y_true, y_pred) == 0.75


def test_log_loss() :
    pass
