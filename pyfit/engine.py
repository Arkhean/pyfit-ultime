"""
useful function for internal purpose only
Also code from pbesquet
Autograd engine implementing reverse-mode autodifferentiation, aka backpropagation.
Heavily inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
"""
# pylint: disable=protected-access
from typing import Union, Tuple, List, Set, Callable, Any
import numpy as np

def as_array(x: Any) -> np.ndarray:
    """
    convert list to ndarray if necessary
    """
    if not isinstance(x, np.ndarray):
        return np.array(x)
    return x

################################################################################


class Tensor:
    """Stores values and their gradients"""

    def __init__(
        self, data: np.ndarray, children: Tuple["Tensor", ...] = (), op: str = ""
    ) -> None:
        self.data: np.ndarray = as_array(data)
        self.grad: np.ndarray = np.zeros(self.data.shape)

        # Internal variables used for autograd graph construction
        self._backward: Callable = lambda: None
        self._prev: Set[Tensor] = set(children)
        self._op = (
            op  # The operation that produced this node, for graphviz / debugging / etc
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        """return new Tensor from slice, this is not a copy"""
        new_tensor = Tensor(self.data[i])
        new_tensor.grad = self.grad[i]
        return new_tensor

    def __add__(self, other: Union["Tensor", np.ndarray]) -> "Tensor":
        _other: Tensor = other if isinstance(other, Tensor) else Tensor(other)
        if self.data.shape == _other.data.shape:
            out: Tensor = Tensor(self.data + _other.data, (self, _other), "+")
            def _backward() -> None:
                self.grad += out.grad  # d(out)/d(self) = 1
                _other.grad += out.grad  # d(out)/d(other) = 1
        elif _other.data.shape == ():
            # addition avec un scalaire
            out = Tensor(self.data + _other.data, (self, _other), "+")
            def _backward() -> None:
                self.grad += out.grad
        else:
            raise Exception(f'bad shapes: {self.data.shape} and {_other.data.shape}')
        out._backward = _backward
        return out

    def __sub__(self, other: Union["Tensor", np.ndarray]) -> "Tensor":
        _other: Tensor = other if isinstance(other, Tensor) else Tensor(other)
        if self.data.shape == _other.data.shape:
            out: Tensor = Tensor(self.data - _other.data, (self, _other), "-")
            def _backward() -> None:
                self.grad += out.grad  # d(out)/d(self) = 1
                _other.grad -= out.grad  # d(out)/d(other) = -1
            out._backward = _backward
            return out
        else:
            raise Exception(f'bad shapes: {self.data.shape} and {_other.data.shape}')

    def __neg__(self) -> "Tensor":
        out: Tensor = Tensor(-self.data, (self,), "neg")
        def _backward() -> None:
            self.grad -= out.grad   # pas sûr ??? TODO
        out._backward = _backward
        return out

    def __mul__(self, other: Union["Tensor", np.ndarray]) -> "Tensor":
        _other = other if isinstance(other, Tensor) else Tensor(other)
        if self.data.shape == _other.data.shape:
            # il faut les mêmes dimensions!
            out = Tensor(self.data * _other.data, (self, _other), "*")
            def _backward() -> None:
                self.grad += out.grad * _other.data  # d(out)/d(self) = other
                _other.grad += out.grad * self.data  # d(out)/d(other) = self
        elif _other.data.shape == ():
            # multiplication par un scalaire
            out = Tensor(self.data * _other.data, (self, _other), "*")
            def _backward() -> None:
                self.grad += out.grad * _other.data  # d(out)/d(self) = other
        else:
            raise Exception(f'bad shapes: {self.data.shape} and {_other.data.shape}')
        out._backward = _backward
        return out

    def __truediv__(self, other: Union["Tensor", np.ndarray]) -> "Tensor":
        _other = other if isinstance(other, Tensor) else Tensor(other)
        if self.data.shape == _other.data.shape:
            # il faut les mêmes dimensions!
            out = Tensor(self.data / _other.data, (self, _other), "/")
            def _backward() -> None:
                self.grad += out.grad / _other.data  # d(out)/d(self) = 1/other
                # d(out)/d(other) = -self/(other*other)
                _other.grad += out.grad * (-self.data / (_other.data * _other.data))
        elif _other.data.shape == ():
            # division par un scalaire
            out = Tensor(self.data / _other.data, (self, _other), "/")
            def _backward() -> None:
                self.grad += out.grad / _other.data  # d(out)/d(self) = 1/other
        else:
            raise Exception(f'bad shapes: {self.data.shape} and {_other.data.shape}')
        out._backward = _backward
        return out

    def __pow__(self, other: int) -> "Tensor":
        if isinstance(other, int):
            out = Tensor(self.data ** other, (self), "**")
            def _backward() -> None:
                self.grad += other * (self.data ** (other - 1))
        else:
            raise Exception('second argument must be an integer')
        out._backward = _backward
        return out

    def dot(self, other: Union["Tensor", np.ndarray]) -> "Tensor":
        _other = other if isinstance(other, Tensor) else Tensor(other)
        if self.data.shape[-1] == _other.data.shape[0]:
            out = Tensor(self.data.dot(_other.data), (self, _other), "dot")
            def _backward() -> None:
                self.grad += _other.data # pas sûr
                _other.grad += self.data
        else:
            raise Exception(f'bad shapes: {self.data.shape} and {_other.data.shape}')
        out._backward = _backward
        return out

    def exp(self) -> "Tensor":
        """Compute exp"""
        out = Tensor(np.exp(self.data), (self,), "exp")
        def _backward() -> None:
            self.grad += out.data
        out._backward = _backward
        return out

    # TODO:
    # def relu(self) -> "Tensor":
    #     """Compute ReLU"""
    #     out = Tensor(0 if self.data < 0 else self.data, (self,), "ReLU")
    #     def _backward() -> None:
    #         self.grad += (out.data > 0) * out.grad
    #     out._backward = _backward
    #     return out

    def backward(self) -> None:
        """Compute gradients through backpropagation"""

        # Topological order all of the children in the graph
        topo: List[Tensor] = []
        visited: Set[Tensor] = set()

        def build_topo(node: Tensor) -> None:
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)

        build_topo(self)

        # Go one variable at a time and apply the chain rule to get its gradient
        self.grad = np.ones((self.data.shape))
        for node in reversed(topo):
            node._backward()

    def __radd__(self, other: Union["Tensor", np.ndarray]) -> "Tensor":
        return self.__add__(other)

    def __rsub__(self, other: Union["Tensor", np.ndarray]) -> "Tensor":
        _other = other if isinstance(other, Tensor) else Tensor(other)
        return _other.__sub__(self)

    def __rmul__(self, other: Union["Tensor", np.ndarray]) -> "Tensor":
        return self.__mul__(other)

    def __rtruediv__(self, other: Union["Tensor", np.ndarray]) -> "Tensor":
        _other = other if isinstance(other, Tensor) else Tensor(other)
        return _other.__truediv__(self)

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, grad={self.grad})"
