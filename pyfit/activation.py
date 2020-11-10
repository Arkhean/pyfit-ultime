"""
Fonctions d'activation utiles aux réseaux de neurones
"""

from typing import Union
import numpy as np

T = Union[float, np.ndarray]

def sigmoid(x: T, deriv: bool = False) -> T:
    """
    sigmoid function 1 / (1+exp(-x))
    """
    if deriv:
        return np.exp(-x) / ((1+np.exp(-x)) ** 2)
    return 1 / (1 + np.exp(-x))

def relu(x: T, deriv: bool = False) -> T:
    """
    relu function max(x, 0)
    """
    if deriv:
        return 1 if x > 0 else 0
    return x if x > 0 else 0

def tanh(x: T, deriv: bool = False) -> T:
    """
    tanh function sinh(x) / cosh(x)
    """
    if deriv:
        return 1 - np.tanh(x) ** 2
    return np.tanh(x)
