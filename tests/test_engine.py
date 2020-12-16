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


def test_tensor_sub() -> None:
    x = Tensor(2.5)
    y = Tensor(3.5)
    z = x - y
    assert z.data == -1
    z.backward()
    assert x.grad == 1  # dz/dx
    assert y.grad == -1  # dz/dy


def test_tensor_div() -> None:
    x = Tensor(1.0)
    y = Tensor(4.0)
    z = x / y
    assert z.data == 0.25
    z.backward()
    assert x.grad == 0.25  # dz/dx
    assert y.grad == -0.0625  # dz/dy


def test_tensor_add() -> None:
    x = Tensor([2, 1])
    y = Tensor([1, 1])
    z = x + y
    assert np.array_equal(z.data, [3, 2])
    z.backward()
    assert np.array_equal(x.grad, [1, 1])
    assert np.array_equal(y.grad, [1, 1])

def test_tensor_mul_scalar() -> None:
    x = Tensor([2, 1])
    z = 2 * x
    assert np.array_equal(z.data, [4, 2])
    z.backward()
    assert np.array_equal(x.grad, [2, 2])

def test_tensor_mul() -> None:
    x = Tensor([2, 1])
    y = Tensor([1, 1])
    z = x * y
    assert np.array_equal(z.data, [2, 1])
    z.backward()
    assert np.array_equal(x.grad, [1, 1])
    assert np.array_equal(y.grad, [2, 1])
