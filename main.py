import numpy as np


class KMeans:
    k: int
    d: int
    centroids: np.ndarray
    clusters: np.ndarray

    def __init__(self, k=3):
        self.k = k

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


if __name__ == "__main__":
    X = np.random.rand(100, 2)

    kmeans = KMeans(k=3)
    kmeans.fit(X)

    print(X)
    print(kmeans.centroids)
