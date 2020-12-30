"""
code from pbesquet and modified
Neural network providing a PyTorch-like API.
Heavily inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/nn.py
"""

import random
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

class Neuron(Module):
    """A single neuron"""

    def __init__(self, in_features: int, activation: str = 'linear'):
        self.w: Tensor = Tensor([random.uniform(-1, 1) for _ in range(in_features)])
        self.b: Tensor = Tensor(0)
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

class Layer(Module):
    """A layer of neurons"""

    def __init__(self, in_features: int, out_features: int, activation: str = 'linear'):
        self.w = Tensor(2 * np.random.random_sample((in_features, out_features)) - 1)
        self.b = Tensor(2 * np.random.random_sample((out_features)) - 1)
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
        return f"Layer of {self.activation} Neurons({len(self.w)})"

class MLP(Module):
    """A Multi-Layer Perceptron, aka shallow neural network"""

    def __init__(self) -> None:
        self.layers: List[Layer] = list()

    def add(self, layer: Layer) -> None:
        # TODO: check dimensions
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
