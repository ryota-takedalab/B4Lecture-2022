import numpy as np


def k_means(data, classes=3):
    length = len(data)
    dimension = data.shape[1]
    
    # pick initial centroids
    # current implement is random
    centroids = np.random.randint(0, length, size=(classes))
    centroids = data[centroids]

    distances = np.zeros((length, classes))

    while True:
        # NOTE: 行列演算に置き換えれたら良いな
        for i in range(length):
            distances[i] = np.sum(np.power(data[i] - centroids, 2), axis=1)
        belongs = np.argmin(distances, axis=1)
        
        centroids_new = np.zeros((classes, dimension))
        for c in range(classes):
            centroids_new[c] = np.average(data[belongs == c], axis=0)

        # judge convergence
        if np.all(centroids_new == centroids):
            break
        centroids = centroids_new
    
    return
