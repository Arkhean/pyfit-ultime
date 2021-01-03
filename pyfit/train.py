"""
Training API
work in progress TODO
"""

# pylint: disable=too-few-public-methods

from typing import Callable, Dict, List
from pyfit.engine import Tensor
from pyfit.nn import Module
from pyfit.optim import Optimizer
from pyfit.data import BatchIterator
from pyfit.metrics import binary_accuracy

# Used to record training history for metrics
History = Dict[str, List[float]]


class Trainer:
    """Encapsulates the model training loop"""

    def __init__(self, model: Module, optimizer: Optimizer, loss: Callable):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss

    def fit(
        self, data_iterator: BatchIterator, num_epochs: int = 500, verbose: bool = False
    ) -> History:
        """Fits the model to the data"""

        history: History = {"loss": [], "acc": []}
        epoch_loss: float = 0
        epoch_acc: float = 0
        epoch_y_true: List[Tensor] = []
        epoch_y_pred: List[Tensor] = []
        for epoch in range(num_epochs):
            # Reset the gradients of model parameters
            self.optimizer.zero_grad()
            # Reset epoch data
            epoch_loss = 0
            epoch_y_true = []
            epoch_y_pred = []

            for batch in data_iterator():
                # Forward pass
                # TODO fix mypy error when mapping model to inputs
                outputs = self.model(batch.inputs)  # type: ignore

                # Loss computation
                batch_loss = self.loss(batch.targets, outputs)
                epoch_loss += batch_loss.data[0, 0]

                # Store batch predictions and ground truth for computing epoch metrics
                epoch_y_pred.extend(outputs)
                epoch_y_true.extend(batch.targets)

                # Backprop and gradient descent
                batch_loss.backward()
                self.optimizer.step()

            # Accuracy computation for epoch
            epoch_acc = binary_accuracy(epoch_y_true, epoch_y_pred)

            # Record training history
            history["loss"].append(epoch_loss)
            history["acc"].append(epoch_acc)

            if verbose and epoch % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], "
                    f"loss: {epoch_loss:.6f}, "
                    f"accuracy: {epoch_acc*100:.2f}%"
                )

        return history
