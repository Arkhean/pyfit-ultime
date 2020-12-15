from pyfit.kmeans import KMeans
import numpy as np


def test_fit():
    # plantera si l'algorithme n'a pas convergé
    X = np.array([[1, 2], [1, 4], [1, 0], [5, 2], [5, 4], [5, 0]])
    kmeans = KMeans(n_clusters=2).fit(X)
    assert np.array_equal(kmeans.labels_, np.array([1, 1, 1, 0, 0, 0])) or np.array_equal(kmeans.labels_, np.array([0, 0, 0, 1, 1, 1]))
    assert np.array_equal(kmeans.centers_, np.array([[5.,  2.], [ 1.,  2.]])) or np.array_equal(kmeans.centers_, np.array([[1.,  2.], [5.,  2.]]))

def test_predict():
    # plantera si l'algorithme n'a pas convergé
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    kmeans = KMeans(n_clusters=2).fit(X)
    y = kmeans.predict([[8,8]])
    assert np.array_equal(y, [[10, 2]])
