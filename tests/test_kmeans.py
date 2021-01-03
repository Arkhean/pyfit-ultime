from pyfit.kmeans import KMeans
from sklearn.cluster import KMeans as sk_KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
import numpy as np


def test_kmeans():
    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    my_kmeans = KMeans(n_clusters = 4).fit(X)
    sk_kmeans = sk_KMeans(n_clusters = 4).fit(X)
    count = dict()
    for y1, y2 in zip(my_kmeans.labels_, sk_kmeans.labels_):
        count[(y1,y2)] = count.get((y1,y2), 0) + 1
    count = list(count.values())
    count = sorted(count, reverse=True)
    assert sum(count[:4])/300 > 0.85



def test_predict():
    # plantera si l'algorithme n'a pas convergé
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    kmeans = KMeans(n_clusters=2).fit(X)
    y = kmeans.predict([[8,8]])
    assert np.array_equal(y, [[10, 2]])
