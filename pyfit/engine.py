"""
useful function for internal purpose only
Also code from pbesquet
Autograd engine implementing reverse-mode autodifferentiation, aka backpropagation.
Heavily inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
"""
# pylint: disable=protected-access
from typing import Union, Tuple, List, Set, Callable, Any
import numpy as np

def as_tensor(x: Any) -> "Tensor":
    """
    convert list to Tensor if necessary
    """
    if not isinstance(x, Tensor):
        return Tensor(x)
    return x

def as_array(x: Any) -> np.ndarray:
    """
    convert list to ndarray if necessary
    """
    if not isinstance(x, np.ndarray):
        return np.array(x)
    return x

################################################################################


class Tensor:
    """
    Stores values and their gradients
    shape must be (batch_size, nb_features)
    input with shape () and (n,) will be convert to (1, n)
    """

    def __init__(
        self, data: np.ndarray, children: Tuple["Tensor", ...] = (), op: str = ""
    ) -> None:
        self.data: np.ndarray = np.atleast_2d(data)
        self.grad: np.ndarray = np.zeros(self.data.shape)
        self.shape = self.data.shape

        # Internal variables used for autograd graph construction
        self._backward: Callable = lambda: None
        self._prev: Set[Tensor] = set(children)
        self._op = (
            op  # The operation that produced this node, for graphviz / debugging / etc
        )

    def zero_grad(self) -> None:
        """Reset gradients for all parameters"""
        self.grad = np.zeros(self.data.shape)

    def __len__(self) -> int:
        """number of sample in tensor"""
        return len(self.data)

    def __getitem__(self, i: Any) -> "Tensor":
        """return new Tensor from slice, this is not a copy"""
        new_tensor = Tensor(self.data[i])
        new_tensor.grad = np.atleast_2d(self.grad[i])
        return new_tensor

    def mean(self) -> "Tensor":
        """compute mean of tensor for each feature (axis=0)"""
        out = Tensor(np.mean(self.data, axis=0), (self,), "mean")
        #print(f"mean {self.shape}")
        def _backward() -> None:
            self.grad += out.grad / ( [self.data.shape[0]] * self.data.shape[1] )
        out._backward = _backward
        return out

    def abs(self) -> "Tensor":
        """compute abs"""
        out = Tensor(self.data if self.data >= 0 else -self.data, (self,), "ReLU")
        def _backward() -> None:
            self.grad += out.grad * (1 if self.data >= 0 else -1) # non dérivable en 0...
        out._backward = _backward
        return out

    def relu(self) -> "Tensor":
        """Compute ReLU"""
        out = Tensor(self.data * (self.data > 0), (self,), "ReLU")
        def _backward() -> None:
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def __add__(self, other: Union["Tensor", np.ndarray, float]) -> "Tensor":
        _other: Tensor = other if isinstance(other, Tensor) else Tensor(other)
        #print(f"add {self.shape} {_other.shape}")
        if self.data.shape == _other.data.shape:
            out = Tensor(self.data + _other.data, (self, _other), "+")
            def _backward() -> None:
                self.grad += out.grad  # d(out)/d(self) = 1
                _other.grad += out.grad  # d(out)/d(other) = 1
        elif self.data.shape[0] > 1 and _other.data.shape[0] == 1:
            out = Tensor(self.data + _other.data, (self, _other), "+")
            def _backward() -> None:
                self.grad += out.grad  # d(out)/d(self) = 1
                axis = tuple(x for x in range(len(_other.shape)) if _other.shape[x] == 1)
                _other.grad += np.sum(out.grad, axis=axis)
        else:
            raise Exception(f'bad shapes: {self.data.shape} and {_other.data.shape}')
        out._backward = _backward
        return out

    def __sub__(self, other: Union["Tensor", np.ndarray, float]) -> "Tensor":
        _other: Tensor = other if isinstance(other, Tensor) else Tensor(other)
        #print(f"sub {self.shape} {_other.shape}")
        return self + (-_other)

    def __neg__(self) -> "Tensor":
        out: Tensor = Tensor(-self.data, (self,), "neg")
        #print(f"neg {self.shape}")
        def _backward() -> None:
            self.grad -= out.grad
        out._backward = _backward
        return out

    def __mul__(self, other: Union["Tensor", np.ndarray, float]) -> "Tensor":
        _other = other if isinstance(other, Tensor) else Tensor(other)
        #print(f"mul: {self.shape} {_other.shape}")
        if self.data.shape == _other.data.shape:
            # il faut les mêmes dimensions!
            out = Tensor(self.data * _other.data, (self, _other), "*")
            def _backward() -> None:
                self.grad += out.grad * _other.data  # d(out)/d(self) = other
                _other.grad += out.grad * self.data  # d(out)/d(other) = self
        elif _other.data.shape == (1, 1):
            # multiplication par un scalaire
            out = Tensor(self.data * _other.data, (self, _other), "*")
            def _backward() -> None:
                self.grad += out.grad * _other.data
        else:
            raise Exception(f'bad shapes: {self.data.shape} and {_other.data.shape}')
        out._backward = _backward
        return out

    def __truediv__(self, other: Union["Tensor", np.ndarray, float]) -> "Tensor":
        _other = other if isinstance(other, Tensor) else Tensor(other)
        #print(f"div {self.shape}")
        if self.data.shape == _other.data.shape:
            # il faut les mêmes dimensions!
            out = Tensor(self.data / _other.data, (self, _other), "/")
            def _backward() -> None:
                self.grad += out.grad / _other.data  # d(out)/d(self) = 1/other
                # d(out)/d(other) = -self/(other*other)
                _other.grad += out.grad * (-self.data / (_other.data * _other.data))
        elif _other.data.shape == (1, 1):
            # division par un scalaire
            out = Tensor(self.data / _other.data, (self, _other), "/")
            def _backward() -> None:
                self.grad += out.grad / _other.data  # d(out)/d(self) = 1/other
        elif self.data.shape == (1, 1):
            # division par un scalaire
            out = Tensor(self.data / _other.data, (self, _other), "/")
            def _backward() -> None:
                _other.grad += out.grad * (-self.data / (_other.data * _other.data))
        else:
            raise Exception(f'bad shapes: {self.data.shape} and {_other.data.shape}')
        out._backward = _backward
        return out

    def __pow__(self, other: int) -> "Tensor":
        #print(f"pow {self.shape}")
        if isinstance(other, int):
            out = Tensor(self.data ** other, (self,), "**")
            def _backward() -> None:
                self.grad += out.grad * other * (self.data ** (other - 1))
        else:
            raise Exception('second argument must be an integer')
        out._backward = _backward
        return out

    def dot(self, other: Union["Tensor", np.ndarray]) -> "Tensor":
        """compute dot product between tensors"""
        _other = other if isinstance(other, Tensor) else Tensor(other)
        #print(f"dot {self.shape} {_other.shape}")
        if self.data.shape[-1] == _other.data.shape[0]:
            out = Tensor(self.data.dot(_other.data), (self, _other), "dot")
            def _backward() -> None:
                self.grad += out.grad.dot(_other.data.T)
                _other.grad += self.data.T.dot(out.grad)
        else:
            raise Exception(f'bad shapes: {self.data.shape} and {_other.data.shape}')
        out._backward = _backward
        return out

    def exp(self) -> "Tensor":
        """Compute exp"""
        out = Tensor(np.exp(self.data), (self,), "exp")
        #print(f"exp {self.shape}")
        def _backward() -> None:
            self.grad += out.grad * out.data
        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        """compute log"""
        out = Tensor(np.log(self.data), (self,), "exp")
        #print(f"log {self.shape}")
        def _backward() -> None:
            self.grad += out.grad / self.data
        out._backward = _backward
        return out

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

    def __radd__(self, other: Union["Tensor", np.ndarray, float]) -> "Tensor":
        return self.__add__(other)

    def __rsub__(self, other: Union["Tensor", np.ndarray, float]) -> "Tensor":
        _other = other if isinstance(other, Tensor) else Tensor(other)
        return _other.__sub__(self)

    def __rmul__(self, other: Union["Tensor", np.ndarray, float]) -> "Tensor":
        return self.__mul__(other)

    def __rtruediv__(self, other: Union["Tensor", np.ndarray, float]) -> "Tensor":
        _other = other if isinstance(other, Tensor) else Tensor(other)
        return _other.__truediv__(self)

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, grad={self.grad})"
