"""
Fonctions d'activation utiles aux réseaux de neurones
"""

from typing import Union
import numpy as np
from pyfit.engine import *

T = Union[float, np.ndarray]

def sigmoid(x: Scalar) -> Scalar:
    """
    sigmoid function 1 / (1+exp(-x))
    """
    return 1 / (1 + (-x).exp())

# def relu(x: Scalar) -> Scalar:
#     """
#     relu function max(x, 0)
#     """
#     if deriv:
#         return 1 if x > 0 else 0
#     return x if x > 0 else 0

def tanh(x: Scalar) -> Scalar:
    """
    tanh function sinh(x) / cosh(x)
    """
    return 2 * sigmoid(2 * x) - 1


activation_functions = {'sigmoid': sigmoid, 'tanh': tanh}
