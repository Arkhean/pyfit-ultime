"""
Decision Tree Classfier class
"""

import numpy as np


class DecisionTreeNode:
    """Node for decision tree"""
    def __init__(self, x: np.ndarray, y: np.ndarray, n_class: int, criterion: str) -> None:
        """

        """
        self.x = x
        self.y = y
        self.n_class = n_class
        self.criterion = criterion

        # number of samples
        self.samples = len(x)
        # number of samples of each class
        self.value = [0] * self.n_class
        for i in range(len(self.y)):
            self.value[self.y[i]] += 1

        # measure of the node impurity.
        if criterion == 'gini':
            self.impurity = self.gini_impurity()

        self.decision_axe = None
        self.decision_value = None
        self.left_node = None
        self.right_node = None

    def gini_impurity(self):
        """
        calculate gini impurity
        G = 1 - sum_{k=1}^K p_{k}^2 where p_{k} is the ratio of class k instances in the node
        G = 0 <=> all samples it applies to belong to the same class ("pure" node)
        """
        self.impurity = 1
        for pk in self.value:
            self.impurity -= (pk/self.samples) ** 2

    def expand(self, max_depth):
        """ expand node by creating 2 children minimizing impurity"""
        # if the node is not pure yet
        if self.impurity != 0:
            # find decision axe and decision_value
            # TODO !
            self.decision_axe = None
            self.decision_value = None

            # separate data
            x_true = list()
            y_ true = list()
            x_false = list()
            y_false = list()
            for x, y in zip(self.x, self.y):
                if x[self.decision_axe] <= self.decision_value:
                    x_true.append(x)
                    y_true.append(y)
                else:
                    x_false.append(x)
                    y_false.append(y)

            # create children
            self.true_node = DecisionTreeNode(x_true, y_true, self.n_class, self.criterion)
            self.false_node = DecisionTreeNode(x_false, y_false, self.n_class, self.criterion)
            # expand(max_depth-1)
            if max_depth != 0:
                self.true_node.expand(max_depth-1)
                self.false_node.expand(max_depth-1)

################################################################################

class DecisionTreeClassifier:
    """

    """
    def __init__(self, criterion: str, max_depth: int) -> None:
        """"""
        self.root = None
        self.criterion = criterion
        self.max_depth = max_depth

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Build a decision tree classifier from the training set (X, y)."""
        # count the number of different classes
        classes = set(y)
        # create root node
        self.root = DecisionTreeNode(x, y, len(classes), 'gini')
        self.root.expand(max_depth)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict class or regression value for X."""
