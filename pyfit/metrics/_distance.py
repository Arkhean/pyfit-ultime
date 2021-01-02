"""
Distance metrics
"""

from typing import Callable
import numpy as np

# Distance function type
Distance = Callable[[np.ndarray, np.ndarray], float]


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Euclidean distance: https://en.wikipedia.org/wiki/Euclidean_distance
    """
    squared_diff: np.ndarray = (a - b) ** 2
    sum_squared_diff: float = np.sum(squared_diff)
    eucl_dist: float = np.sqrt(sum_squared_diff)
    return eucl_dist
