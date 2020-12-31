"""
code from pbesquet and modified
Neural network providing a PyTorch-like API.
Heavily inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/nn.py
"""

from typing import List
import numpy as np
from pyfit.engine import Tensor
from pyfit.activation import ACTIVATION_FUNCTIONS


class Module:
    """A differentiable computation"""

    def zero_grad(self) -> None:
        """Reset gradients for all parameters"""

        for p in self.parameters():
            p.grad = np.zeros((p.grad.shape))

    def parameters(self) -> List[Tensor]:
        """Return parameters"""

        raise NotImplementedError

################################################################################

class Neuron(Module):
    """A single neuron"""

    def __init__(self, in_features: int, activation: str = 'linear'):
        self.w: Tensor = Tensor(2 * np.random.random_sample((in_features, 1)) - 1)
        self.b: Tensor = Tensor(2 * np.random.random_sample((1)) - 1)
        self.nonlin = activation != 'linear'
        self.activation = ACTIVATION_FUNCTIONS[activation]

    def __call__(self, x: Tensor) -> Tensor:
        act: Tensor = x.dot(self.w) + self.b
        if self.nonlin and self.activation is not None:
            return self.activation(act)
        return act

    def parameters(self) -> List[Tensor]:
        return [self.w, self.b]

    def __repr__(self) -> str:
        return f"{self.activation} Neuron({len(self.w)})"

################################################################################

class Layer(Module):
    """A layer of neurons"""

    def __init__(self, in_features: int, out_features: int, activation: str = 'linear'):
        self.in_features = in_features
        self.out_features = out_features
        self.nonlin = activation != 'linear'
        self.activation = ACTIVATION_FUNCTIONS[activation]

    def __call__(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def parameters(self) -> List[Tensor]:
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError

################################################################################

class Dense(Layer):
    """A layer of neurons"""

    def __init__(self, in_features: int, out_features: int, activation: str = 'linear'):
        Layer.__init__(self, in_features, out_features, activation)
        self.w = Tensor(2 * np.random.random_sample((in_features, out_features)) - 1)
        self.b = Tensor(2 * np.random.random_sample((out_features)) - 1)

    def __call__(self, x: Tensor) -> Tensor:
        act: Tensor = x.dot(self.w) + self.b
        if self.nonlin and self.activation is not None:
            return self.activation(act)
        return act

    def parameters(self) -> List[Tensor]:
        return [self.w, self.b]

    def __repr__(self) -> str:
        return f"Layer of {self.activation} Neurons({len(self.w)})"

################################################################################

class Dropout(Layer):
    """A Dropout layer"""

    def __init__(self, in_features: int, rate: float):
        Layer.__init__(self, in_features, in_features, 'linear')
        self.rate = rate
        self._mask = None

    def __call__(self, x: Tensor) -> Tensor:
        self._mask = np.random.uniform(size=x.shape) > self.rate
        y = x * self._mask
        return y

    def parameters(self) -> List[Tensor]:
        return []

    def __repr__(self) -> str:
        return f"Layer of Dropout: rate = {self.rate}"

################################################################################

class Activation(Layer):
    """An activation layer"""

    def __init__(self, in_features: int, activation: str = 'linear'):
        Layer.__init__(self, in_features, in_features, activation)

    def __call__(self, x: Tensor) -> Tensor:
        if self.nonlin and self.activation is not None:
            return self.activation(x)
        return x

    def parameters(self) -> List[Tensor]:
        return []

    def __repr__(self) -> str:
        return f"Layer of Activation: {self.activation}"

################################################################################

class Model(Module):
    """A Multi-Layer Perceptron, aka shallow neural network"""

    def __init__(self) -> None:
        self.layers: List[Layer] = list()

    def add(self, layer: Layer) -> None:
        """add layer to the model"""
        if len(self.layers) == 0 or self.layers[-1].out_features == layer.in_features:
            self.layers.append(layer)

    def __call__(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[Tensor]:
        res = list()
        for layer in self.layers:
            res += layer.parameters()
        return res

    def __repr__(self) -> str:
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
