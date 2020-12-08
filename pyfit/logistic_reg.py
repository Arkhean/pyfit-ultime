"""
logistic regression algorithm
"""
import numpy as np
from pyfit.activation import sigmoid

class LogisticReg:
# Descente de gradient:
#on cherche à minimiser la fonction coût à l'aide de Theta --> logistique?
    def __init__(self,
                x_train: np.ndarray,
                y_train: np.ndarray,
                learning_rate=0.01,
                max_iter=20) -> None:
        x_train_t = np.transpose(x_train)
        #add column
        x_train_1 = self.passage_x1(x_train)
        self.theta = np.random.random((len(x_train[0])+1, 1))
        nb_iter = 1
        inter = sigmoid(np.matmul(x_train_1, self.theta)) - y_train
        grad_theta = 2 / len(x_train[0]) * np.matmul(x_train_t, inter)
        while(nb_iter < max_iter and grad_theta[0] < 10):
            inter = sigmoid(np.matmul(x_train_1, self.theta)) - y_train
            grad_theta = 2 / len(x_train[0]) * np.matmul(x_train_t, inter)
            self.theta = self.theta - learning_rate * grad_theta
            nb_iter += 1

    def passage_x1(self, x_entree: np.ndarray) -> np.ndarray:
        x_1 = []
        for ligne in x_entree:
            ligne_bis = ligne
            ligne_bis = np.append(ligne_bis, 1)
            x_1.append(ligne_bis)
        x_1 = np.array(x_1)
        return x_1

    def predict(self, x_entree: np.ndarray) -> np.ndarray:
        x_entree_1 = self.passage_x1(x_entree)
        y_pred_value = np.matmul(x_entree_1, self.theta)
        y_pred = np.zeros(len(y_pred_value), 1)
        for i, element in enumerate(y_pred):
            if element >= 0.5:
                y_pred[i][0] = 1
        return y_pred
