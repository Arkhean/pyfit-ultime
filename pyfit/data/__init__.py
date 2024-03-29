"""
data processing functions
"""

# pylint: disable=too-few-public-methods

from typing import Any, List, Tuple, NamedTuple, Iterator
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pyfit.engine import as_array
from pyfit.engine import Tensor


def train_test_split(*arrays: Any, **options: Any) -> List[Any]:
    """
    split data between train and test set with ratio: test_size=0.2
    """
    if len(arrays) == 0:
        raise ValueError("At least one array required as input")
    test_size = options.get('test_size', 0.2) # default is 0.2

    indices = np.random.permutation(len(arrays[0]))
    split_index = int((1 - test_size) * len(arrays[0]))
    training_idx, test_idx = indices[:split_index], indices[split_index:]
    res = list()
    for array in arrays:
        train_set, test_set = array[training_idx], array[test_idx]
        res += [train_set, test_set]
    return res

def make_classification(
        nb_samples: int,
        nb_class: int,
        nb_features: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    generate clusters of points normally distributed
    """
    # Initialisation x et y
    x_points = np.zeros((nb_samples, nb_features))
    y_points = np.zeros((nb_samples, 1))
    lim_plan = [0, 6] # on pose ça pour changer plus tard?
    nb_sample_class = (int)(nb_samples / nb_class)
    for i in range(nb_class):
        #print(i)
        #On choisi un centre de cluster aléatoirement
        centre = list()
        for _ in range(nb_features):
            centre.append(random.randint(lim_plan[0] * 10, lim_plan[1] * 10) / 10)
        lower_lim = np.arange(0, nb_sample_class) / nb_sample_class
        upper_lim = np.arange(1, nb_sample_class + 1) / nb_sample_class
        points = np.random.uniform(low=lower_lim, high=upper_lim,
                                        size=[nb_features, nb_sample_class]).T
        np.random.shuffle(points[:, 1])
        # On place les points dans l'espace approprié
        for p, point in enumerate(points):
            for j in range(nb_features):
                x_points[i * nb_sample_class + p][j] = point[j] + centre[j]
            y_points[i * nb_sample_class + p] = i
    return x_points, y_points

def plot_data(x: np.ndarray, y: np.ndarray) -> None:
    """Plot some 2D data"""

    _, ax = plt.subplots()
    scatter = ax.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower right", title="Classes")
    ax.add_artist(legend1)
    plt.xlim((min(x[:, 0]) - 0.1, max(x[:, 0]) + 0.1))
    plt.ylim((min(x[:, 1]) - 0.1, max(x[:, 1]) + 0.1))



################################################################################

Batch = NamedTuple("Batch", [("inputs", List[Tensor]), ("targets", List[Tensor])])

class BatchIterator:
    """Iterates on data by batches"""

    def __init__(
        self,
        inputs: List[Tensor],
        targets: List[Tensor],
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> None:
        self.inputs: List[Tensor] = inputs
        self.targets: List[Tensor] = targets
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self) -> Iterator[Batch]:
        starts = list(range(0, len(self.inputs), self.batch_size))
        if self.shuffle:
            random.shuffle(starts)

        for start in starts:
            end = start + self.batch_size
            batch_inputs = self.inputs[start:end]
            batch_targets = self.targets[start:end]
            yield Batch(batch_inputs, batch_targets)
