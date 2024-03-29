"""
Optimization algorithms for gradient descent
"""

# pylint: disable=too-few-public-methods

from typing import List
from pyfit.engine import Tensor


class Optimizer:
    """Base class for optimizers"""

    def __init__(self, parameters: List[Tensor]):
        self.parameters: List[Tensor] = parameters

    def zero_grad(self) -> None:
        """Reset gradients for all parameters"""

        for p in self.parameters:
            p.zero_grad()

    def step(self) -> None:
        """Take a step of gradient descent"""

        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer"""

    def __init__(self, parameters: List[Tensor], learning_rate: float = 0.01):
        super().__init__(parameters)
        self.learning_rate: float = learning_rate

    def step(self) -> None:
        """Update model parameters in the opposite direction of their gradient"""

        for p in self.parameters:
            p.data -= self.learning_rate * p.grad
