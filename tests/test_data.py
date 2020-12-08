"""
test on data functions and classes
"""
from pyfit.data.preprocessing import *
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


def test_one_hot_encoding_1():
    enc = OneHotEncoder()
    X = ['house', 'flat', 'flat', 'house', 'house', 'flat', 'van']
    enc.fit(X)
    pred = enc.transform(X)
    true = [[1,0,0], [0,1,0], [0,1,0], [1,0,0], [1,0,0], [0,1,0], [0,0,1]]
    assert np.array_equal(pred, true)


def test_one_hot_encoding_2():
    enc = OneHotEncoder()
    X = [['Male', 1], ['Female', 3], ['Female', 2]]
    enc.fit(X)
    pred = enc.transform([['Female', 1], ['Male', 3]])
    true = np.array([[0, 1, 1, 0, 0], [1, 0, 0, 1, 0]])
    assert np.array_equal(pred, true)
