"""
logistic regression algorithm
"""
import numpy as np
from pyfit.activation import sigmoid, softmax
from engine import passage_x1
from preprocessing import OneHotEncoder

class LogisticReg:
    """
    Implementation of logistic regression
    """
    def __init__(self,
                x_train: np.ndarray,
                y_train: np.ndarray,
                learning_rate=0.01,
                max_iter=100,
                nb_class=2) -> None:
        """
        Coefficients initialization
        """
        if(nb_class == 2):
            m = len(x_train[0])
            x_train_1 = passage_x1(x_train)
            x_train_t = np.transpose(x_train_1)
            x_train_t = np.transpose(x_train_1)
            self.theta = np.random.random((m+1, 1))
            nb_iter = 1
            inter = sigmoid(np.matmul(x_train_1, self.theta)) - y_train
            grad_theta = (2 / m) * np.matmul(x_train_t, inter)
            while nb_iter < max_iter:
                inter = sigmoid(np.matmul(x_train_1, self.theta)) - y_train
                grad_theta = (2 / m) * np.matmul(x_train_t, inter)
                self.theta = self.theta - learning_rate * grad_theta
                nb_iter += 1
        else:
            # On One Hot Encode les classes:
            self.theta = np.zeros([x.shape[1],len(np.unique(y))])
            m = x_train.shape[0]
            enc = OneHotEncoder()
            enc.fit(y_train)
            y = enc.transform(y_train)
            for i in range(max_iter):
                scores = np.dot(x_train,weights) #Then we compute raw class scores given our input and current weights
                prob = softmax(scores) #Next we perform a softmax on these scores to get their probabilities
                grad = (-1 / m) * np.dot(x_train.T,(y - prob)) + self.theta
                self.theta = self.theta - (learningRate * grad)

    def predict(self, x_entree: np.ndarray) -> np.ndarray:
        """
        Predict outputs
        """
        x_entree_1 = passage_x1(x_entree)
        y_pred_value = np.matmul(x_entree_1, self.theta)
        y_pred = y_pred_value
        for i, element in enumerate(y_pred):
            if element >= 0.5:
                y_pred[i][0] = 1
        return y_pred
