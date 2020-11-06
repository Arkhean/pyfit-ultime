# TODO
import numpy as np
import math
# MSE

def mean_squared_error(y_truth,y_pred):
    n=len(y_pred)
    mse= n* (np.mean(np.array([y_truth,y_pred])))**2
    return mse

# MAE

def mean_absolute_error(y_truth,y_pred):
    return(np.mean(np.array([y_truth,y_pred])))


# log loss
def log_loss(y_truth,y_pred):
    n=len(y_pred)
    sum=0
    for i,yi in enumerate(y_truth):
        sum+= yi*math.log(y_pred[i]) + (i-yi)*math.log(1-pred[i])
    logl=(-1/n)*sum
    return logl