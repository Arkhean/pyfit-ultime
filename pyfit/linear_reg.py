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
        x_train_1 = self.passage_x1(x_train)
        self.theta = np.random.random((len(x_train[0])+1, 1))
        nb_iter = 1
        inter = np.matmul(x_train_1, self.theta) - y_train
        grad_theta = 2 / len(x_train[0]) * np.matmul(x_train_t, inter)
        while(nb_iter < max_iter and grad_theta[0] < 10):
            inter = np.matmul(x_train_1, self.theta) - y_train
            grad_theta = 2 / len(x_train[0]) * np.matmul(x_train_t, inter)
            self.theta = self.theta - learning_rate * grad_theta
            nb_iter += 1
    
    def passage_x1(self,x_entree: np.ndarray) -> np.ndarray:
        x_1 = []
        for ligne in x_entree:
            ligne_bis = ligne
            ligne_bis = np.append(ligne_bis,1)
            x_1.append(ligne_bis)
        x_1=np.array(x_1)
        return(x_1)

    def predict(self, x_entree: np.ndarray) -> np.ndarray:
        x_entree_1 = self.passage_x1(x_entree)
        y_pred = np.matmul(x_entree_1, self.theta)
        return y_pred
