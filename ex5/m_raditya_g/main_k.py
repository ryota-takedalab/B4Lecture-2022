import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from k_cluster import Kmeans
import argparse


def main():
    # Argparse
    parser = argparse.ArgumentParser(description='Name of the Audio File')
    parser.add_argument('-fn', metavar='-f', dest='filename', type=str, help='Enter the Audio File Name',
                        required=True)
    parser.add_argument('-n', metavar='-nc', dest='cluster_number', type=int,
                        help='Enter Cluster Number', default=4, required=False)
    parser.add_argument('-i', metavar='-mi', dest='max_iteration', type=int,
                        help='Enter Max Iteration', default=100, required=False)
    parser.add_argument('-mode', metavar='-m', dest='kmeans_mode', type=str,
                        help='Enter Kmeans Mode ("LGB", "random" or "k++")', default='random', required=False)
    args = parser.parse_args()

    # Kmeans
    data = pd.read_csv(args.filename)
    fig = plt.figure(figsize=(8, 8))
    if len(data.columns) == 2:
        kmeans = Kmeans(data=np.array([data["x"], data["y"]]).T, init=args.kmeans_mode,
                        n_clusters=args.cluster_number, max_iter=args.max_iteration, random_state=10)
        cluster, centroids, n= kmeans.cluster()
        ax = fig.add_subplot()
        for cl in cluster:
            ax.scatter(cl[:, 0], cl[:, 1])
        ax.scatter(centroids[:, 0], centroids[:, 1], linewidth=3, marker="x",
                   color="k", label="centroid")
        ax.set_title(f"{args.filename}, k={args.cluster_number}, "
                     f"n_iter={n}, method={args.kmeans_mode}")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        plt.savefig(f"{args.filename}_{args.kmeans_mode}.png")
        plt.show()
        plt.close()
    elif len(data.columns) == 3:
        kmeans = Kmeans(data=np.array([data["x"], data["y"], data["z"]]).T, init='LBG', n_clusters=4,
                        max_iter=10, random_state=10)
        cluster, centroids, n = kmeans.cluster()
        ax = fig.add_subplot(projection='3d')
        for cl in cluster:
            ax.plot(cl[:, 0], cl[:, 1], cl[:, 2],
                    marker="o", markerfacecolor="None",
                    linestyle="None")
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
                   linestyle="None", marker="x", color="k",
                   label="centroid")
        ax.set_title(f"{args.filename}, k={args.cluster_number}, "
                     f"n_iter={n}, method={args.kmeans_mode}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.savefig(f"{args.filename}_{args.kmeans_mode}.png")
        plt.show()
        plt.close()


if __name__ == '__main__':
    main()
