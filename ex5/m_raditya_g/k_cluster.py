import numpy as np
import random


# Trying to use class
class Kmeans:
    def __init__(self, data, init, n_clusters, max_iter, random_state):
        """
        :param data (np.ndarray): Observed Data
        :param init (str): Kmeans method
        :param n_clusters (int): Number of Cluster
        :param max_iter (int): Maximum number of Iteration
        :param random_state (int): Random seed
        """
        self.mode = init
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.data = data
        self.random_state = random_state

    def random_init(self):
        """
        Random Method Initialization
        :return: cluster (np.ndarray): Array of cluster
        :return: centroids (np.ndarray): Array of Centroids Location
        """
        np.random.seed(self.random_state)
        index = random.sample(range(self.data.shape[0]), self.n_clusters)
        centroids = self.data[index]
        # K-means
        cluster, centroid, n = self.kmeans(centroids)
        return cluster, centroid, n

    def k_plus_plus_init(self):
        """
        K++ Initialization
        :return: cluster (np.ndarray): Array of cluster
        :return: centroids (np.ndarray): Array of Centroids Location
        """
        np.random.seed(self.random_state)
        index = random.sample(range(self.data.shape[0]), self.n_clusters)
        centroid = self.data[index]
        for k in range(self.n_clusters):
            # Calculate distances from points to the centroids
            dist = np.sum((self.data - centroid[k, :]) ** 2, axis=1)
            # Normalize the distances
            dist /= np.sum((self.data - centroid[k, :]) ** 2)
            # Choose remaining points based on their distances
            centroid[k, :] = self.data[np.random.choice(range(len(self.data)), size=1, p=dist)]
        cluster, centroids, n = self.kmeans(centroid)
        return cluster, centroids, n

    def lbg_init(self):
        """
        LBG Method Initialization
        :return: cluster (np.ndarray): Array of cluster
        :return: centroids (np.ndarray): Array of Centroids Location
        """
        # initialize
        centroids = np.array([np.mean(self.data, axis=0)])
        delta = np.full(self.data.shape[1], 0.01)
        cluster = []
        n = 0
        while len(centroids) < self.n_clusters:
            # calculate the centroids of clusters
            centroids = np.append(centroids - delta, centroids + delta, axis=0)
            # K-means
            cluster, centroids, n = self.kmeans(centroids)
        # Select centroids from the initialized lbg
        centroids = centroids[random.sample(range(centroids.shape[0]), self.n_clusters)]
        return cluster, centroids, n

    def kmeans(self, centroids):
        """
        K-means Clustering
        :return: cluster (np.ndarray): Array of cluster
        :return: centroids (np.ndarray): Array of Centroids Location
        """
        # K-means
        cluster = []
        temp = centroids.copy()
        temp[0, 0] += 1
        n = 0
        while np.allclose(centroids, temp, rtol=1e-05, atol=1e-08) is False \
                and n < self.max_iter:
            n += 1
            temp = centroids.copy()
            # calculate the square of distance between x and centroid
            sq_of_dist = np.array(
                [np.sum((self.data - centroid) ** 2, axis=1) for centroid in centroids])
            # clustering
            class_x = np.argmin(sq_of_dist, axis=0)
            cluster = [self.data[class_x == c] for c in range(len(centroids))]
            # update
            centroids = np.array([np.mean(cl, axis=0) for cl in cluster])
        print(n)
        return cluster, centroids, n

    def cluster(self):
        if self.mode == 'random':
            cluster, centroid, n = self.random_init()

        elif self.mode == 'LBG':
            cluster, centroid, n = self.lbg_init()

        elif self.mode == 'k++':
            cluster, centroid, n = self.k_plus_plus_init()

        else:
            raise Exception("Choose between 'random','LBG' and 'k++'")
        return cluster, centroid, n
