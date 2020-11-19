"""
linear regression algorithm
"""
import numpy as np

class LinearReg:
    def __init__(self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            learning_rate=0.2,
            max_iter=20) -> None:
        x_train_t = np.transpose(x_train)
        #add column
        x_train_1 = []
        for ligne in x_train:
            ligne_bis = ligne
            ligne_bis.append(1)
            x_train_1.append(ligne_bis)
        self.theta = np.random.random((len(x_train[0]), 1))
        nb_iter = 1
        inter = np.matmul(x_train_1, self.theta) - y_train
        grad_theta = 2 / len(x_train[0]) * np.matmul(x_train_t, inter)
        while(nb_iter < max_iter and grad_theta[0] < 10):
            inter = np.matmul(x_train_1, self.theta) - y_train
            grad_theta = 2 / len(x_train[0]) * np.matmul(x_train_t, inter)
            self.theta = self.theta - learning_rate * grad_theta
            nb_iter += 1

    def predict(self, x_entree: np.ndarray) -> np.ndarray:
        #je crois qu'on a une issue avec theta0 qui est le biais et doit être multiplié par 1
        x_entree_1 = []
        for ligne in x_entree:
            ligne_bis = ligne
            ligne_bis.append(1)
            x_entree_1.append(ligne_bis)
        y_pred = np.matmul(x_entree_1, self.theta)
        return y_pred
