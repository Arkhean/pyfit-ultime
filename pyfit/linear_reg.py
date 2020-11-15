"""
linear regression algorithm
"""
import numpy as np

class Linear_reg:
    def __init__(self, x_train, y_train, analytical=False,learning_rate=0.2,max_iter=10):
        x_train_t = np.transpose(x_train)
        #add column
        x_train_1 = []
        for i in range(len(x_train)):
            ligne = x_train[i]
            ligne.append(1)
            x_train_1.append(ligne)
        #analytical approach
        if analytical:
            self.theta= np.matmul(np.matmul(np.linalg.inv(np.matmul(x_train_t, x_train)), x_train_t), y_train)
        #descente de gradient --> nécessite loss function et dérivée
        else:
            #génération aléatoire de théta
            self.theta = np.random.random((len(x_train[0]), 1))
            for _ in range(max_iter):
                inter = np.matmul(x_train_1, self.theta) - y_train
                grad_theta = 2 / len(x_train[0]) * np.matmul(x_train_t, inter)
                self.theta += learning_rate * grad_theta
                print("gradient")
                print(grad_theta)

    def prevision(self, x_entree):
        #je crois qu'on a une issue avec theta0 qui est le biais et doit être multiplié par 1
        x_entree_1 = []
        for i in range(len(x_entree)):
            ligne = x_entree[i]
            ligne.append(1)
            x_entree_1.append(ligne)
        print("theta final")
        print(self.theta)
        y_pred = np.matmul(x_entree_1, self.theta)
        return(y_pred)

# regression à 1 param
x_train = [[1],
        [2],
        [4],
        [6]]
y_train = [[1],
            [2],
            [4],
            [6]]
model=Linear_reg(x_train,y_train,learning_rate=0.2,max_iter=10)

x_entree = [[3],
            [4],
            [5]]
y_pred=model.prevision(x_entree)
print("y")
print(y_pred)

# x1= [1,2,3,4]
# x2=[[1], [2], [3], [4]]

# [result]= np.matmul(x1,x2)
# print(result)
