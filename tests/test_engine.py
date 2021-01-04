"""
Unit tests for autodiff engine
"""

# pylint: disable=missing-docstring

from pyfit.engine import Tensor
import numpy as np

def test_tensor() -> None:
    x = Tensor(1.0)
    y = (x * 2 + 1)
    assert y.data == 3
    y.backward()
    assert x.grad == 2  # dy/dx

################################################################################

def test_tensor_add_tensor() -> None:
    x = Tensor([2, 1])
    y = Tensor([1, 1])
    z = x + y
    assert np.array_equal(z.data, [[3, 2]])
    z.backward()
    assert np.array_equal(x.grad, [[1, 1]])
    assert np.array_equal(y.grad, [[1, 1]])

def test_tensor_add_scalar() -> None:
    x = Tensor([2, 1])
    y = x + 1
    assert np.array_equal(y.data, [[3, 2]])
    y.backward()
    assert np.array_equal(x.grad, [[1, 1]])
    x = Tensor([2, 1])
    z = 1 + x
    assert np.array_equal(z.data, [[3, 2]])
    z.backward()
    assert np.array_equal(x.grad, [[1, 1]])

def test_tensor_add() -> None:
    x = Tensor([1, 1, 2])
    y = Tensor([[.5, .5, .5], [.3, .2, .2]])
    z = x + y
    assert np.array_equal(z.data, [[1.5, 1.5, 2.5], [1.3, 1.2, 2.2]])
    z.backward()
    assert np.array_equal(x.grad, [[2, 2, 2]])
    assert np.array_equal(y.grad, [[1, 1, 1], [1, 1, 1]])

################################################################################

def test_tensor_sub_tensor() -> None:
    x = Tensor(2.5)
    y = Tensor(3.5)
    z = x - y
    assert z.data == -1
    z.backward()
    assert x.grad == 1  # dz/dx
    assert y.grad == -1  # dz/dy

def test_tensor_sub_scalar() -> None:
    x = Tensor([2, 1])
    y = x - 1
    assert np.array_equal(y.data, [[1, 0]])
    y.backward()
    assert np.array_equal(x.grad, [[1, 1]])
    z = 1 - x
    assert np.array_equal(z.data, [[-1, 0]])
    z.backward()
    assert np.array_equal(x.grad, [[0, 0]])

################################################################################

def test_tensor_mul_tensor() -> None:
    x = Tensor([2, 1])
    y = Tensor([1, 1])
    z = x * y
    assert np.array_equal(z.data, [[2, 1]])
    z.backward()
    assert np.array_equal(x.grad, [[1, 1]])
    assert np.array_equal(y.grad, [[2, 1]])

def test_tensor_mul_scalar() -> None:
    x = Tensor([2, 1])
    y = 2 * x
    assert np.array_equal(y.data, [[4, 2]])
    y.backward()
    assert np.array_equal(x.grad, [[2, 2]])
    z = x * 3
    assert np.array_equal(z.data, [[6, 3]])

################################################################################

def test_tensor_div_tensor() -> None:
    x = Tensor(1.0)
    y = Tensor(4.0)
    z = x / y
    assert z.data == 0.25
    z.backward()
    assert x.grad == 0.25  # dz/dx
    assert y.grad == -0.0625  # dz/dy

def test_tensor_div_scalar() -> None:
    x = Tensor([2, 1])
    y = x / 2
    assert np.array_equal(y.data, [[1, 0.5]])
    y.backward()
    assert np.array_equal(x.grad, [[0.5, 0.5]])
    z = 1 / x
    assert np.array_equal(z.data, [[0.5, 1]])

################################################################################

def test_tensor_dot() -> None:
    x = Tensor([1, 1])
    a = Tensor([[1, 2], [3, 4]])
    y = x.dot(a)
    assert np.array_equal(y.data, [[4, 6]])

################################################################################

def test_tensor_pow_scalar() -> None:
    x = Tensor([2, 1])
    y = x ** 2
    assert np.array_equal(y.data, [[4, 1]])

################################################################################

def test_tensor_getitem() -> None:
    x = Tensor([1, 0, 3, 2, -1, 5, -2])
    y = x[0, 2] # les tensors sont 2D pour des raisons pratiques...
    assert np.array_equal(y.data, [[3]])
    y = x[0, 3:5]
    assert np.array_equal(y.data, [[2, -1]])

################################################################################

def test_tensor_relu() -> None:
    x = Tensor([1, 0, 3, 2, -1, 5, -2])
    y = x.relu()
    assert np.array_equal(y.data, [[1, 0, 3, 2, 0, 5, 0]])
