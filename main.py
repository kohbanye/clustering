from abc import ABCMeta, abstractmethod
import numpy as np


class Clustering(metaclass=ABCMeta):
    k: int
    d: int
    centroids: np.ndarray
    clusters: np.ndarray

    def __init__(self, k: int):
        self.k = k

    @abstractmethod
    def fit(self, X: np.ndarray):
        pass


class KMeans(Clustering):
    def fit(self, X: np.ndarray):
        self.d = X.shape[1]
        initial_centroids_index = np.random.choice(X.shape[0], self.k, replace=False)
        self.centroids = X[initial_centroids_index, :]

        n = X.shape[0]
        self.clusters = np.zeros(n)
        while True:
            new_clusters = np.zeros(n)
            for i, x in enumerate(X):
                distances = np.linalg.norm(x - self.centroids, axis=1)
                new_clusters[i] = np.argmin(distances)

            if np.array_equal(self.clusters, new_clusters):
                break

            self.clusters = new_clusters
            for i in range(self.k):
                self.centroids[i] = np.mean(X[self.clusters == i], axis=0)


class KMeansPP(Clustering):
    def distance(self, x: np.ndarray):
        distances = np.linalg.norm(x - self.centroids, axis=1)
        return np.min(distances)

    def fit(self, X: np.ndarray):
        self.d = X.shape[1]

        first_centroid_index = np.random.choice(X.shape[0], 1)
        self.centroids = X[first_centroid_index, :]
        for _ in range(self.k - 1):
            distances = np.array([self.distance(x) for x in X])
            probabilities = distances**2 / np.sum(distances**2)
            new_centroid_index = np.random.choice(X.shape[0], 1, p=probabilities)
            self.centroids = np.append(self.centroids, X[new_centroid_index, :], axis=0)

        n = X.shape[0]
        self.clusters = np.zeros(n)
        while True:
            new_clusters = np.zeros(n)
            for i, x in enumerate(X):
                distances = np.linalg.norm(x - self.centroids, axis=1)
                new_clusters[i] = np.argmin(distances)

            if np.array_equal(self.clusters, new_clusters):
                break

            self.clusters = new_clusters
            for i in range(self.k):
                self.centroids[i] = np.mean(X[self.clusters == i], axis=0)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    generator = np.random.default_rng(0)
    X = np.concatenate(
        [
            generator.normal(loc=(0, 0), scale=0.5, size=(100, 2)),
            generator.normal(loc=(4, 2), scale=0.5, size=(100, 2)),
            generator.normal(loc=(2, 5), scale=0.5, size=(100, 2)),
            generator.normal(loc=(5, 7), scale=0.5, size=(100, 2)),
            generator.normal(loc=(7, 4), scale=0.5, size=(100, 2)),
        ]
    )

    kmeans = KMeans(5)
    kmeans_pp = KMeansPP(5)
    kmeans.fit(X)
    kmeans_pp.fit(X)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].scatter(X[:, 0], X[:, 1], c=kmeans.clusters)
    ax[0].scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c="red")
    ax[0].set_title("KMeans")
    ax[1].scatter(X[:, 0], X[:, 1], c=kmeans_pp.clusters)
    ax[1].scatter(kmeans_pp.centroids[:, 0], kmeans_pp.centroids[:, 1], c="red")
    ax[1].set_title("KMeans++")
    plt.show()

    print(kmeans.centroids)
    print(kmeans_pp.centroids)
