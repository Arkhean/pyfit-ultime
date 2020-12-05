"""
data processing functions
"""
from typing import Any, List
import numpy as np


def train_test_split(*arrays: Any, **options: Any) -> List[Any]:
    """
    split data between train and test set with ratio
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

def one_hot_encode(x: np.ndarray) -> np.ndarray:
    """
    transform categorical data to vector with one one and zeros
    """
    categories = dict()
    index = 0
    for element in x:
        if element not in categories:
            categories[x] = index
            index += 1
    cat_count = len(categories)
    res = np.zeros((len(x), cat_count))
    for i in range(len(x)):
        res[i, categories[x]] = 1
    return res

def make_classification(n_samples: int, nb_class: int, nb_features):
    """
    generate clusters of points normally distributed
    """
    # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html
    # https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/datasets/samples_generator.py#L37
    # Initialisation x et y
    x_points = np.zeros((n_samples, n_features))
    y_points = np.zeros((n_samples, d_type=np.int))
    # x_points = []
    # y_points = []
    # Chaque classe est une répartission dans un hypercube de dim nb_sample --> On choisi taille 1x1x1x1...
    # Un cluster par classe
    lim_plan= [0,6] # on pose ça pour changer plus tard?
    # size = lim_plan[1] - lim_plan[0]
    nb_sample_class = n_samples/nb_class
    centres_cluster=[]
    for i in range(nb_class):
        #On choisi un centre de cluster aléatoirement des centres?
        centre =[]
        for _ in range(nb_features):
            centre.append(random.randint(lim_plan[0]*10,lim_plan[1]*10)/10)
        centre.append()
        lower_lim = numpy.arange(0,nb_sample_class)/nb_sample_class
        upper_lim = numpy.arange(1,nb_sample_class+1)/nb_sample_class
        points = numpy.random.uniform(low = lower_lim, high = upper_limits, size = [nb_features, nb_sample_class]).T
        numpy.random.shuffle(points[:,1])
        # On place les points dans l'espace approprié
        for p,point in enumerat(points):
            for j in range(nb_features):
                x_points[p][j] = point[j] + centre[j]
            y_points[p] = p
    
    return x_points,y_points
    

    

class Scaler:
    """
    Standardize features by removing the mean and scaling to unit variance
    """
    def __init__(self) -> None:
        pass

    def fit(self, x: np.ndarray) -> None:
        """
        Compute the mean and std to be used for later scaling.
        """

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Perform standardization by centering and scaling
        """

