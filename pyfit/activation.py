"""
Fonctions d'activation utiles aux réseaux de neurones
"""

from typing import Union
import numpy as np
from pyfit.engine import Tensor

T = Union[Tensor, np.ndarray]

def sigmoid(x: Tensor) -> Tensor:
    """
    sigmoid function 1 / (1+exp(-x))
    """
    if not isinstance(x, Tensor):
        return 1 / (1 + np.exp(-x))
    return 1 / (1 + (-x).exp())

def relu(x: Tensor) -> Tensor:
    """
    relu function max(x, 0)
    """
    if not isinstance(x, Tensor):
        return x if x > 0 else 0
    return x.relu()

def tan_h(x: Tensor) -> Tensor:
    """
    tanh function sinh(x) / cosh(x)
    """
    if not isinstance(x, Tensor):
        return np.tanh(x)
    return 2 * sigmoid(2 * x) - 1


ACTIVATION_FUNCTIONS = {
    'sigmoid': sigmoid,
    'tan_h': tan_h,
    'relu': relu,
    'linear': None
}

################################################################################

def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
    return sm

def getProbsAndPreds(someX):
    probs = softmax(np.dot(someX,w))
    preds = np.argmax(probs,axis=1)
    return probs,preds