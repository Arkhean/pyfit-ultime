"""
linear regression algorithm
"""
import numpy as np
from pyfit.engine import passage_x1


class LinearReg:
    """
    Implementation of linear regression
    """
    def __init__(self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            learning_rate=0.2,
            max_iter=20,
            analytical=False) -> None:
        """
        Coefficients initialization
        """
        m = len(x_train[0])
        x_train_1 = passage_x1(x_train)
        x_train_t = np.transpose(x_train_1)
        self.theta = np.random.random((m + 1, 1))
        print(self.theta.shape)
        if analytical:
            theta_inter = np.matmul(x_train_t, x_train_1)
            theta_inter2 = np.matmul(np.linalg.inv(theta_inter), x_train_t)
            self.theta = np.matmul(theta_inter2, y_train)
        else:
            nb_iter = 1
            inter = np.matmul(x_train_1, self.theta) - y_train
            grad_theta = 2 / len(x_train[0]) * np.matmul(x_train_t, inter)
            while nb_iter < max_iter:
                inter = np.matmul(x_train_1, self.theta) - y_train
                grad_theta = 2 / len(x_train[0]) * np.matmul(x_train_t, inter)
                self.theta = self.theta - learning_rate * grad_theta
                nb_iter += 1

    def predict(self, x_entree: np.ndarray) -> np.ndarray:
        """
        Predict outputs
        """
        x_entree_1 = passage_x1(x_entree)
        y_pred = np.matmul(x_entree_1, self.theta)
        return y_pred
