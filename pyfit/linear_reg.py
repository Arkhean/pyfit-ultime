"""
linear regression algorithm
"""
import numpy as np

class Linear_reg:
    def __init__(self, x_train, y_train, analytical=False,learning_rate=0.2,max_iter=10):
        x_train_t = np.transpose(x_train)
        new_row = [1 for _ in range(len(x_train[0]))]
        x_train_1= np.vstack((x_train,new_row))
        #analytical approach
        if analytical:
            self.theta= np.matmul(np.matmul(np.linalg.inv(np.matmul(x_train_t, x_train)), x_train_t), y_train)
        #descente de gradient --> nécessite loss function et dérivée
        else:
            #génération aléatoire de théta
            self.theta=np.random.random((len(y_train)+1, 1))

            for _ in range(max_iter):
                grad_theta = 2 / len(x_train[0]) * np.matmul(x_train_t, np.matmul(x_train_1, self.theta) - y_train)
                self.theta -= learning_rate * grad_theta
        print(self.theta)
    
    def prevision(self, x_entree):
        #je crois qu'on a une issue avec theta0 qui est le biais et doit être multiplié par 1
        new_row=[1 for _ in range(len(x_entree[0]))]
        x_entree_1= np.vstack((x_entree,new_row))
        y_pred = np.matmul(self.theta, x_entree_1)
        return(y_pred)
    

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

print(y_pred)
