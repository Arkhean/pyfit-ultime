from pyfit.linear_reg import Linear_reg
import numpy as np

def test_1D_lin_reg():
    x_train = np.array([[1], [2], [4], [6]])
    y_train = np.array[[1], [2], [4], [6]])
    model=Linear_reg(x_train,y_train,learning_rate=0.01,max_iter=10)
    x_entree = np.array([[3], [4], [5]])
    y_pred=model.prevision(x_entree)