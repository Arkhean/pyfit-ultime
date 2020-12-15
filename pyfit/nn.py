"""
code from pbesquet and modified
Neural network providing a PyTorch-like API.
Heavily inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/nn.py
"""

import random
from typing import List
from pyfit.engine import Tensor
from pyfit.activation import *


class Module:
    """A differentiable computation"""

    def zero_grad(self) -> None:
        """Reset gradients for all parameters"""

        for p in self.parameters(): # TODO:
            p.grad = 0

    def parameters(self) -> Tensor:
        """Return parameters"""

        raise NotImplementedError

# TODO: activations...
class Neuron(Module):
    """A single neuron"""

    def __init__(self, in_features: int, activation: str = 'linear'):
        self.w: Tensor = Tensor([random.uniform(-1, 1) for _ in range(in_features)])
        self.b: Tensor = Tensor(0)
        self.nonlin = activation != 'linear'
        self.activation = ACTIVATION_FUNCTIONS[activation]

    def __call__(self, x: Tensor) -> Tensor:
        act: Tensor = x * sel.w + self.b
        if self.nonlin:
            return self.activation(act)
        return act

    def parameters(self) -> Tensor:
        return self.w + [self.b] # TODO: concat?, référence et numpy ???

    def __repr__(self) -> str:
        return f"{self.activation} Neuron({len(self.w)})"

class Layer(Module):
    """A layer of neurons"""

    def __init__(self, in_features: int, out_features: int, activation: str = 'linear'):
        self.w = Tensor(2 * np.random.randomm_sample((in_features, out_features)) - 1)
        self.b = Tensor(2 * np.random.randomm_sample((out_features)) - 1)
        self.nonlin = activation != 'linear'
        self.activation = ACTIVATION_FUNCTIONS[activation]

    def __call__(self, x: Tensor) -> Tensor:
        act: Tensor = x.dot(sel.w) + self.b
        if self.nonlin:
            return self.activation(act)
        return act

    def parameters(self) -> Tensor:
        # TODO garder les références!
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self) -> str:
        return f"Layer of {self.activation} Neurons({len(self.w)})"

# TODO: add layers
class MLP(Module):
    """A Multi-Layer Perceptron, aka shallow neural network"""

    def __init__(self, input_features: int, layers: List[int]):
        sizes: List[int] = [input_features] + layers
        self.layers = [
            Layer(sizes[i], sizes[i + 1], nonlin=i != len(layers) - 1)
            for i in range(len(layers))
        ]

    def __call__(self, x: Vector) -> Vector:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> Vector:
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self) -> str:
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
