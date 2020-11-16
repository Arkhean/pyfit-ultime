"""
linear regression algorithm
"""
import numpy as np

class Linear_reg:
    def __init__(self, x_train, y_train, analytical = False, learning_rate = 0.2, max_iter = 20):
        x_train_t = np.transpose(x_train)
        #add column
        x_train_1 = []
        for i in range(len(x_train)):
            ligne = x_train[i]
            ligne.append(1)
            x_train_1.append(ligne)
        #analytical approach --> pas valable dans beaucoup de cas, on la laisse de côté
        if analytical:
            inter_1 = np.linalg.inv(np.matmul(x_train_t, x_train))
            inter_2 = np.matmul(inter_1, x_train_t)
            self.theta= np.matmul(inter_2, y_train)
        #descente de gradient --> nécessite loss function et dérivée
        else:
            #génération aléatoire de théta
            self.theta = np.random.random((len(x_train[0]), 1))
            nb_iter = 1
            inter = np.matmul(x_train_1, self.theta) - y_train #dif avec l'obtenue
            grad_theta = 2 / len(x_train[0]) * np.matmul(x_train_t, inter)
            while(nb_iter<max_iter and grad_theta[0]<10):
            # for _ in range(max_iter):
                inter = np.matmul(x_train_1, self.theta) - y_train #dif avec l'obtenue
                grad_theta = 2 / len(x_train[0]) * np.matmul(x_train_t, inter)
                self.theta = self.theta - learning_rate * grad_theta
                nb_iter += 1

    def predict(self, x_entree):
        #je crois qu'on a une issue avec theta0 qui est le biais et doit être multiplié par 1
        x_entree_1 = []
        for i in range(len(x_entree)):
            ligne = x_entree[i]
            ligne.append(1)
            x_entree_1.append(ligne)
        y_pred = np.matmul(x_entree_1, self.theta)
        return(y_pred)
