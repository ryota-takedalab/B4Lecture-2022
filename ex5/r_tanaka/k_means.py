import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image


def k_means(X, k, max_iter=300):
    """k-means algorithm

    Args:
        X (np.ndarray): input data
        k (int): number of cluster
        max_iter (int, optional): maximum of iteration. Defaults to 300.

    Returns:
        np.ndarray: the data of clusters
        np.ndarray: the data of centroids
    """
    X_size, n_features = X.shape

    # randomly initialize initial centroids
    centroids = X[np.random.choice(X_size, k)]
    # array for the new centroids
    new_centroids = np.zeros((k, n_features))
    # array to store the cluster information to which each data belongs
    clusters = np.zeros(X_size)

    for _ in range(max_iter):
        # loop for all input data
        for i in range(X_size):
            # calculate the distance to each centroid from the data
            distances = np.sum((centroids - X[i]) ** 2, axis=1)
            # update the cluster based on distances
            clusters[i] = np.argsort(distances)[0]

        # recalculate centroid for all clusters
        for j in range(k):
            new_centroids[j] = X[clusters == j].mean(axis=0)

        # break if centrois has not changed
        if np.sum(new_centroids == centroids) == k:
            print("break")
            break
        centroids = new_centroids

    return clusters, centroids


def render_frame(fig, ax, angle):
    """convert a 3D scatter plot into a PIL Image

    Args:
        fig (matplotlib.figure.Figure): figure data
        ax (list): list of axes objects
        angle (int): angle of rotation

    Returns:
        list: PIL Image
    """
    ax.view_init(30, angle)
    plt.close()
    # PIL Image に変換
    buf = BytesIO()
    fig.savefig(buf, bbox_inches='tight', pad_inches=0.0)
    return Image.open(buf)


def main():
    # read csv files as DataFrame
    data1 = pd.read_csv("data1.csv")
    data2 = pd.read_csv("data2.csv")
    data3 = pd.read_csv("data3.csv")

    # set drawing area
    plt.rcParams["figure.figsize"] = (10, 10)

    # change DataFrame form to array form
    data1_array = data1.to_numpy()
    data2_array = data2.to_numpy()
    data3_array = data3.to_numpy()

    # apply k-means algorithm to data1
    k = 4
    clusters, centroids = k_means(data1_array, k)
    data1["class"] = clusters
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    ax1.scatter(data1["x"], data1["y"], c=data1["class"])
    ax1.scatter(centroids[:, 0], centroids[:, 1], s=300, marker='*', color='red', label="centroids")
    ax1.set(title="data1, k=%i" % k, xlabel="x", ylabel="y")
    ax1.legend()
    fig1.savefig("data1_k=%i.png" % k)

    # apply k-means algorithm to data2
    k = 2
    clusters, centroids = k_means(data2_array, k)
    data2["class"] = clusters
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.scatter(data2["x"], data2["y"], c=data2["class"])
    ax2.scatter(centroids[:, 0], centroids[:, 1], s=300, marker='*', color='red', label="centroids")
    ax2.set(title="data2, k=%i" % k, xlabel="x", ylabel="y")
    ax2.legend()
    fig2.savefig("data2_k=%i.png" % k)

    # apply k-means algorithm to data3
    k = 3
    clusters, centroids = k_means(data3_array, k)
    data3["class"] = clusters
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(projection='3d')
    ax3.scatter(data3["x"], data3["y"], data3["z"], c=data3["class"])
    ax3.scatter(centroids[:, 0], centroids[:, 1], s=300, marker='*', color='red', label="centroids")
    ax3.set(title="data3, k=%i" % k, xlabel="x", ylabel="y", zlabel="z")
    ax3.legend()
    images = [render_frame(fig3, ax3, angle) for angle in range(360)]
    images[0].save('data3_k=%i.gif' % k, save_all=True, append_images=images[1:], duration=100, loop=0)


if __name__ == "__main__":
    main()
