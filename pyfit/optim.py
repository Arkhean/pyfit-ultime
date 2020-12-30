"""
Optimization algorithms for gradient descent
"""

# pylint: disable=too-few-public-methods

from pyfit.engine import Tensor
from typing import List


class Optimizer:
    """Base class for optimizers"""

    def __init__(self, parameters: List[Tensor]):
        self.parameters: List[Tensor] = parameters

    def zero_grad(self) -> None:
        """Reset gradients for all parameters"""

        for p in self.parameters:
            p.zero_grad()


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer"""

    def __init__(self, parameters: List[Tensor], learning_rate: float = 0.01):
        super().__init__(parameters)
        self.learning_rate: float = learning_rate

    def step(self) -> None:
        """Update model parameters in the opposite direction of their gradient"""

        for p in self.parameters:
            print(f"optimizer : {p.grad}")
            p.data -= self.learning_rate * p.grad
