from pyfit.linear_reg import LinearReg
import numpy as np

def test_1D_lin_reg():
    x_train = np.array([[1], [2], [4], [6]])
    y_train = np.array([[1], [2], [4], [6]])
    model=LinearReg(x_train,y_train,learning_rate=0.01,max_iter=30)
    x_entree = np.array([[3], [4], [5]])
    y_pred=model.predict(x_entree)
    for i,element in enumerate(y_pred):
        assert element[0] > x_entree[i][0] - 0.2
        assert element[0] < x_entree[i][0] + 0.2

def test_1D_lin_reg_analytic():
    x_train = np.array([[1], [2], [4], [6]])
    y_train = np.array([[1], [2], [4], [6]])
    model=LinearReg(x_train, y_train, analytical = True)
    x_entree = np.array([[3], [4], [5]])
    y_pred=model.predict(x_entree)
    for i,element in enumerate(y_pred):
        assert element[0] > x_entree[i][0] - 0.2
        assert element[0] < x_entree[i][0] + 0.2

# Test reg linéaire en plus de dimensions.
def test_2D_lin_reg():
    x_train = np.array([[1,1], [2,2], [4,4], [6, 6]])
    y_train = np.array([[1,1], [2,2], [4,4], [6,6]])
    model=LinearReg(x_train,y_train,learning_rate=0.01,max_iter=30)
    x_entree = np.array([[3,3], [4,4], [5,5]])
    y_pred=model.predict(x_entree)
    for i,element in enumerate(y_pred):
        assert element[0] > x_entree[i][0] - 0.2
        assert element[0] < x_entree[i][0] + 0.2

def test_2D_lin_reg_analytic():
    x_train = np.array([[1,1], [2,2], [4,4], [6, 6]])
    y_train = np.array([[1,1], [2,2], [4,4], [6,6]])
    model=LinearReg(x_train,y_train,learning_rate=0.01,max_iter=30)
    x_entree = np.array([[3,3], [4,4], [5,5]])
    y_pred=model.predict(x_entree)
    for i,element in enumerate(y_pred):
        assert element[0] > x_entree[i][0] - 0.2
        assert element[0] < x_entree[i][0] + 0.2