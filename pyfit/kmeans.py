"""
Kmeans clustering algorithm
"""
from typing import List
import numpy as np


class KMeans:
    """
    Kmeans class, access labels_ and centers_ as with scikit-learn API
    """

    def __init__(self, n_clusters: int) -> None:
        """
        create class ready to fit
        """
        self.n_clusters = n_clusters
        self.centers_: List[np.ndarray] = list() # TODO mieux ?
        self.labels_: List[int] = list()

    def fit(self, dataset: np.ndarray, max_iter: int = 50) -> KMeans:
        """
        Compute k-means clustering.
        """
        # Initialize centroids by choosing randomly k points from the dataset
        n_points, _ = dataset.shape
        centroids = dataset[np.random.randint(n_points, size=self.n_clusters), :]
        iterations = 0
        old_centroids = None
        labels = np.zeros(len(dataset))

        # while the maximum number of iterations is not reached and while the centroids change
        while not (iterations == max_iter and np.array_equal(centroids, old_centroids)):
            old_centroids = centroids
            iterations += 1

            # The label is the one of the closest centroid
            for i, p in enumerate(dataset):
                dist_vec = np.zeros((len(centroids), 1))
                for j, c in enumerate(centroids):
                    dist_vec[j] = np.linalg.norm(p-c)
                labels[i] = np.argmin(dist_vec) #indice de la valeur minimum

            # Each centroid is the geometric mean
            # of the points that have that centroid's label
            for j in range(self.n_clusters):
                group = [i for i in range(len(dataset)) if labels[i] == j]
                if len(group) != 0:
                    centroids[j, :] = np.mean(dataset[group], axis=0)
                else:
                    centroids[j, :] = dataset[np.random.randint(len(dataset))]
        print("nb iterations : "+str(iterations))

        self.centers_ = centroids
        self.labels_ = labels
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the closest cluster each sample in X belongs to.
        """
