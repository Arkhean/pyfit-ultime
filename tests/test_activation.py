from pyfit.activation import *
from pyfit.engine import Tensor
from math import *

def test_sigmoid():
    x = Tensor([1, 3, 4])
    x_result = [1, 3, 4]
    y = sigmoid(x)
    y_result = []
    for x_i in x_result:
        y_result.append(1/(1+exp(-x_i)))
    assert y.data[0][0] == y_result[0]
    assert y.data[0][1] == y_result[1]
    assert y.data[0][2] == y_result[2]

def test_relu():
    x = Tensor([-1, 1, 3])
    y = relu(x)
    y_result = [0, 1, 3]
    assert y.data[0][0] == y_result[0]
    assert y.data[0][1] == y_result[1]
    assert y.data[0][2] == y_result[2]

def test_tan_h():
    x = Tensor([1, 3, 4])
    x_result = [1, 3, 4]
    y = tan_h(x)
    y_result = []
    epsilon = 0.000001
    for x_i in x_result:
        y_result.append(tanh(x_i))
    assert y.data[0][0] < y_result[0] + epsilon
    assert y.data[0][0] > y_result[0] - epsilon
    assert y.data[0][1] < y_result[1] + epsilon
    assert y.data[0][1] > y_result[1] - epsilon
    assert y.data[0][2] < y_result[2] + epsilon
    assert y.data[0][2] > y_result[2] - epsilon
