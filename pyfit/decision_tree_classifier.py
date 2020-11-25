"""
Decision Tree Classfier class
inspiration: http://www.oranlooney.com/post/ml-from-scratch-part-4-decision-tree/
"""

from typing import Tuple, List, Optional
import numpy as np

def best_split_point(x: np.ndarray, y: np.ndarray, column: int, n_class: int) -> Tuple[float, float, int]:
    """
    internal fonction, find best split value for column, minimizing gini impurity
    """
    # sorting y by the values of x makes it almost trivial to count classes for
    # above and below any given candidate split point.
    ordering = np.argsort(x[:, column])
    classes = y[ordering]
    # these vectors tell us how many of each class are present "below" (to the
    # left) of any given candidate split point.
    class_below = [(classes == n).cumsum() for n in range(n_class)]
    # Subtracting the cummulative sum from the total gives us the reversed
    # cummulative sum. These are how many of each class are above (to the
    # right) of any given candidate split point.
    # Because class_0_below is a cummulative sum the last value in the array is
    # the total sum. That means we don't need to make another pass through the
    # array just to get the total; we can just grab the last element.
    class_above = [c[-1] - c for c in class_below]

    below_total = np.arange(1, len(y)+1)
    above_total = np.arange(len(y)-1, -1, -1)

    # gini = sum_{i=1}^C p(i)(1-p(i))
    # we can now calculate Gini impurity in a vectorized operation.
    below_gini = np.zeros((len(classes),))
    for i in range(n_class):
        tmp = np.sum(class_below[:i], axis=0) + np.sum(class_below[i+1:], axis=0)
        below_gini += class_below[i] * tmp
    below_gini /= (below_total ** 2)

    above_gini = np.zeros((len(classes),))
    for i in range(n_class):
        tmp = np.sum(class_above[:i], axis=0) + np.sum(class_above[i+1:], axis=0)
        above_gini += class_above[i] *tmp
    above_gini /= (above_total ** 2)

    # last is divided by 0 so NaN
    above_gini[-1] = 1

    gini = below_gini + above_gini

    # we need to reverse the above sorting to get the rule into the form
    # C_n < split_value.
    best_split_rank = np.argmin(gini)
    best_split_gini = gini[best_split_rank]
    best_split_index = np.argwhere(ordering == best_split_rank).item(0)
    best_split_value = x[best_split_index, column]

    return best_split_gini, best_split_value, column

class DecisionTreeNode:
    """Node for decision tree"""
    def __init__(self, x: np.ndarray, y: np.ndarray, n_class: int, criterion: str) -> None:
        """
        initialize node and calculate impurity
        """
        self.x: np.ndarray = x
        self.y: np.ndarray = y
        self.n_class: int = n_class
        self.criterion: str = criterion
        self.is_leaf: bool = True

        # number of samples
        self.samples: int = len(x)
        if self.samples == 0:
            raise Exception("node without data, x is empty")
        # number of samples of each class
        self.value: List[int] = [0] * self.n_class
        for i in range(len(self.y)):
            self.value[self.y[i]] += 1

        # measure of the node impurity.
        if criterion == 'gini':
            self.gini_impurity()

        if self.impurity == 0:
            self.is_leaf = True

        self.decision_axe: int = -1
        self.decision_value: float = 0
        self.left_node: Optional[DecisionTreeNode] = None
        self.right_node: Optional[DecisionTreeNode] = None

    def gini_impurity(self) -> None:
        """
        calculate gini impurity
        G = 1 - sum_{k=1}^K p_{k}^2 where p_{k} is the ratio of class k instances in the node
        G = 0 <=> all samples it applies to belong to the same class ("pure" node)
        """
        self.impurity: float = 1
        for n_samples in self.value:
            self.impurity -= (n_samples/self.samples) ** 2

    def expand(self, max_depth: Optional[int]) -> None:
        """ expand node by creating 2 children minimizing impurity"""
        # if the node is not pure yet
        if self.impurity != 0:
            # find decision axe and decision_value
            splits = [best_split_point(self.x, self.y, column, self.n_class)
                                        for column in range(self.x.shape[1])]
            splits.sort()
            _, split_point, column = splits[0]
            self.decision_axe = column
            self.decision_value = split_point
            # index for spliting data
            below = self.x[:, column] <= split_point
            above = self.x[:, column] > split_point
            # create children
            self.left_node = DecisionTreeNode(self.x[below], self.y[below],
                                                    self.n_class, self.criterion)
            self.right_node = DecisionTreeNode(self.x[above], self.y[above],
                                                    self.n_class, self.criterion)
            self.is_leaf = False
            # expand(max_depth-1)
            if max_depth is not None and max_depth != 0:
                self.left_node.expand(max_depth-1)
                self.right_node.expand(max_depth-1)
            else:
                self.left_node.expand(None)
                self.right_node.expand(None)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        predict class for x
        """
        if self.left_node is not None and self.right_node is not None:
            if self.is_leaf:
                return np.argmax(self.value)
            if x[self.decision_axe] <= self.decision_value:
                return self.left_node.predict(x)
            return self.right_node.predict(x)
        raise RuntimeError("method fit has not been called yet")


################################################################################

class DecisionTreeClassifier:
    """ A decision tree classifier, parameters:
    - criterion: 'gini' only
    - max_depth: bound for the tree depth
    """
    def __init__(self, criterion: str = 'gini', max_depth: int = None) -> None:
        """  """
        self.root: Optional[DecisionTreeNode] = None
        self.criterion: str = criterion
        self.max_depth: Optional[int] = max_depth

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Build a decision tree classifier from the training set (X, y)."""
        # count the number of different classes
        classes = set(y)
        # create root node
        self.root = DecisionTreeNode(x, y, len(classes), 'gini')
        self.root.expand(self.max_depth)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict class or regression value for X."""
        if self.root is not None:
            return self.root.predict(x)
        raise RuntimeError("method fit has not been called yet")
