"""
linear regression algorithm
"""
import numpy as np

class Linear_reg:
    def __init__(self, x_true, y_true, analytical=False,learning_rate=0.2,maxit=10):
        x_true_t = np.transpose(x_true)
        #analytical approach
        if analytical:
            self.theta= np.matmul(np.matmul(np.linalg.inv(np.matmul(x_true_t, x_true)), x_true_t), y_true)
        #descente de gradient --> nécessite loss function et dérivée
        else:
            #génération aléatoire de théta
            self.theta=np.random.random((len(y_true)+1, 1))
            for _ in range(maxit):
                grad_theta = 2 / len(x_true[0]) * np.matmul(x_true_t,mat.matmul(x_true,theta)-y_true)
                theta -= learning_rate * grad_theta
    
    def prevision(self, x_entree):
        #je crois qu'on a une issue avec theta0 qui est le biais et doit être multiplié par 1
        y_pred = np.matmul(self.theta, x_entree)
        return(y_pred)