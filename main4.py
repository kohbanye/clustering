from abc import ABCMeta, abstractmethod
import numpy as np


class Clustering(metaclass=ABCMeta):
    k: int
    d: int
    centroids: np.ndarray
    clusters: np.ndarray
    first_centroids: np.ndarray
    potential: np.ndarray

    def __init__(self, k: int):
        self.k = k
        self.potential = np.empty(0)

    @abstractmethod
    def fit(self, X: np.ndarray):
        pass


class KMeans(Clustering):
    def fit(self, X: np.ndarray):
        self.d = X.shape[1]
        initial_centroids_index = np.random.choice(X.shape[0], self.k, replace=False)
        self.centroids = X[initial_centroids_index, :]
        self.first_centroids = X[initial_centroids_index, :]

        n = X.shape[0]
        self.clusters = np.zeros(n)
        epoch = 0
        old_sum = -1
        while True:
            new_clusters = np.zeros(n)
            sum_distance = 0.0
            for i, x in enumerate(X):
                distances = np.linalg.norm(x - self.centroids, axis=1)
                new_clusters[i] = np.argmin(distances)
                sum_distance += np.min(distances)

            tmp_index = np.arange(self.centroids.shape[0])
            xx, yy = np.meshgrid(tmp_index, tmp_index)
            distances = np.linalg.norm(self.centroids[xx]-self.centroids[yy], axis=2)
            self.potential = np.append(self.potential, np.max(distances))

            if abs((old_sum-sum_distance)/sum_distance) < 0.0000001:
                break
            old_sum = sum_distance

            ##if np.array_equal(self.clusters, new_clusters):
            ##    break

            self.clusters = new_clusters
            for i in range(self.k):
                cluster = X[self.clusters == i, :]
                ##tmp_index = np.arange(cluster.shape[0])
                ##xx, yy = np.meshgrid(tmp_index, tmp_index)
                ##distances = np.linalg.norm(cluster[xx]-cluster[yy], axis=2)
                ##max_distance =  np.max(distances)
                ##if potential < max_distance:
                ##    potential = max_distance
                if cluster.shape[0] > 0:
                    self.centroids[i] = np.mean(cluster, axis=0)
            
            ##self.potential = np.append(self.potential, potential)

            epoch += 1
            if epoch >= 1000:
                break


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
            self.first_centroids = np.append(
                self.centroids, X[new_centroid_index, :], axis=0
            )

        n = X.shape[0]
        self.clusters = np.zeros(n)
        epoch = 0
        old_sum = -1
        while True:
            new_clusters = np.zeros(n)
            sum_distance = 0.0
            for i, x in enumerate(X):
                distances = np.linalg.norm(x - self.centroids, axis=1)
                new_clusters[i] = np.argmin(distances)
                sum_distance += np.min(distances)

            tmp_index = np.arange(self.centroids.shape[0])
            xx, yy = np.meshgrid(tmp_index, tmp_index)
            distances = np.linalg.norm(self.centroids[xx]-self.centroids[yy], axis=2)
            self.potential = np.append(self.potential, np.max(distances))

            if abs((old_sum-sum_distance)/sum_distance) < 0.0000001:
                break
            old_sum = sum_distance

            ##if np.array_equal(self.clusters, new_clusters):
            ##    break

            potential = 0.0
            self.clusters = new_clusters
            for i in range(self.k):
                cluster = X[self.clusters == i, :]
                ##tmp_index = np.arange(cluster.shape[0])
                ##xx, yy = np.meshgrid(tmp_index, tmp_index)
                ##distances = np.linalg.norm(cluster[xx]-cluster[yy], axis=2)
                ##max_distance =  np.max(distances)
                ##if potential < max_distance:
                ##    potential = max_distance
                if cluster.shape[0] > 0:
                    self.centroids[i] = np.mean(cluster, axis=0)
            
            ##self.potential = np.append(self.potential, potential)
            
            epoch += 1
            if epoch >= 1000:
                break


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from dataset import get_data

    X = get_data("wineqr")

    # k is determined by the elbow method
    k = 10

    kmeans = KMeans(k)
    kmeans_pp = KMeansPP(k)
    kmeans.fit(X)
    kmeans_pp.fit(X)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(np.arange(1, kmeans.potential.shape[0] + 1), kmeans.potential)
    ax[0].set_title("potential of k-means")
    ax[1].plot(np.arange(1, kmeans_pp.potential.shape[0] + 1), kmeans_pp.potential)
    ax[1].set_title("potential of k-means++")
    plt.show()

    print("Initial potential")
    print("\tk-means:", kmeans.potential[0])
    print("\tk-means++:", kmeans_pp.potential[0])
    print("Final potential")
    print("\tk-means:", kmeans.potential[-1])
    print("\tk-means++:", kmeans_pp.potential[-1])
