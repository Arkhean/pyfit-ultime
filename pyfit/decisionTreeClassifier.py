"""
Decision Tree Classfier class
inspiration: http://www.oranlooney.com/post/ml-from-scratch-part-4-decision-tree/
"""

import numpy as np


def best_split_point(X, y, column):
    # sorting y by the values of X makes it almost trivial to count classes for
    # above and below any given candidate split point.
    ordering = np.argsort(X[:,column])
    classes = y[ordering]

    # these vectors tell us how many of each class are present "below" (to the
    # left) of any given candidate split point.
    class_0_below = (classes == 0).cumsum()
    class_1_below = (classes == 1).cumsum()

    # Subtracting the cummulative sum from the total gives us the reversed
    # cummulative sum. These are how many of each class are above (to the
    # right) of any given candidate split point.
    #
    # Because class_0_below is a cummulative sum the last value in the array is
    # the total sum. That means we don't need to make another pass through the
    # array just to get the total; we can just grab the last element.
    class_0_above = class_0_below[-1] - class_0_below
    class_1_above = class_1_below[-1] - class_1_below

    # below_total = class_0_below + class_1_below
    below_total = np.arange(1, len(y)+1)
    # above_total = class_0_above + class_1_above
    above_total = np.arange(len(y)-1, -1, -1)

    # we can now calculate Gini impurity in a single vectorized operation.
    # The naive formula would be:
    #     (class_1_below/below_total)*(class_0_below/below_total)
    # however, divisions are expensive and we can get this down to only one
    # division if we combine the denominator term.
    gini = class_1_below * class_0_below / (below_total ** 2) + \
           class_1_above * class_0_above / (above_total ** 2)

    gini[np.isnan(gini)] = 1

    # we need to reverse the above sorting to get the rule into the form
    # C_n < split_value.
    best_split_rank = np.argmin(gini)
    best_split_gini = gini[best_split_rank]
    best_split_index = np.argwhere(ordering == best_split_rank).item(0)
    best_split_value = X[best_split_index, column]

    return best_split_gini, best_split_value, column

def best_split_point2(X, y, column, n_class):
    ordering = np.argsort(X[:,column])
    classes = y[ordering]
    class_below = [(classes == n).cumsum() for n in range(n_class)]
    class_above = [c[-1] - c for c in class_below]

    below_total = np.arange(1, len(y)+1)
    above_total = np.arange(len(y)-1, -1, -1)

    gini = np.prod(class_below, axis=0) / (below_total ** n_class) + \
           np.prod(class_above, axis=0) / (above_total ** n_class)

    gini[np.isnan(gini)] = 1
    best_split_rank = np.argmin(gini)
    best_split_gini = gini[best_split_rank]
    best_split_index = np.argwhere(ordering == best_split_rank).item(0)
    best_split_value = X[best_split_index, column]

    return best_split_gini, best_split_value, column

class DecisionTreeNode:
    """Node for decision tree"""
    def __init__(self, x: np.ndarray, y: np.ndarray, n_class: int, criterion: str) -> None:
        """
        initialize node and calculate impurity
        """
        self.x = x
        self.y = y
        self.n_class = n_class
        self.criterion = criterion
        self.is_leaf = True

        # number of samples
        self.samples = len(x)
        if self.samples == 0 :
            raise Exception("node without data, x is empty")
        # number of samples of each class
        self.value = [0] * self.n_class
        for i in range(len(self.y)):
            self.value[self.y[i]] += 1

        # measure of the node impurity.
        if criterion == 'gini':
            self.gini_impurity()

        if self.impurity == 0:
            self.is_leaf = True

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
            print(f"impurity: {self.impurity}")
            # find decision axe and decision_value
            splits = [ best_split_point2(self.x, self.y, column, self.n_class) for column in range(self.x.shape[1]) ]
            splits.sort()
            gini, split_point, column = splits[0]
            self.decision_axe = column
            self.decision_value = split_point
            # index for spliting data
            below = self.x[:,column] <= split_point
            above = self.x[:,column] > split_point
            # create children
            self.left_node = DecisionTreeNode(self.x[below], self.y[below], self.n_class, self.criterion)
            self.right_node = DecisionTreeNode(self.x[above], self.y[above], self.n_class, self.criterion)
            self.is_leaf = False
            # expand(max_depth-1)
            if max_depth != 0:
                self.left_node.expand(max_depth-1)
                self.right_node.expand(max_depth-1)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.is_leaf:
            return np.argmax(self.value)
        if x[self.decision_axe] <= self.decision_value:
            return self.left_node.predict(x)
        return self.right_node.predict(x)


################################################################################

class DecisionTreeClassifier:
    """  """
    def __init__(self, criterion: str='gini', max_depth: int=None) -> None:
        """  """
        self.root = None
        self.criterion = criterion
        self.max_depth = max_depth

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Build a decision tree classifier from the training set (X, y)."""
        # count the number of different classes
        classes = set(y)
        # create root node
        self.root = DecisionTreeNode(x, y, len(classes), 'gini')
        self.root.expand(self.max_depth)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict class or regression value for X."""
        return self.root.predict(x)
