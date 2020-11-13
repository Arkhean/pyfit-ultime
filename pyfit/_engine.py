"""
useful function for internal purpose only
"""
import numpy as np

def as_array(x):
    """
    convert list to ndarray if necessary
    """
    if not x.isinstance(np.ndarray):
        return np.array(x)
    return x
