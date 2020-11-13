"""
useful function for internal purpose only
"""
from typing import Any
import numpy as np

def as_array(x: Any) -> np.ndarray:
    """
    convert list to ndarray if necessary
    """
    if not isinstance(x, np.ndarray):
        return np.array(x)
    return x
