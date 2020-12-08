"""
test on data functions and classes
"""
from pyfit.data import *
import numpy as np


def test_scaler():
    data = [[0, 0], [0, 0], [1, 1], [1, 1]]
    scaler = Scaler()
    scaler.fit(data)
    assert np.array_equal(scaler.mean_, [0.5, 0.5])
    x_scaled = scaler.transform(data)
    assert np.array_equal(x_scaled, [[-1, -1], [-1, -1], [1, 1], [1, 1]])
    x_other = scaler.transform([[2, 2]])
    assert np.array_equal(x_other, [[3, 3]])
