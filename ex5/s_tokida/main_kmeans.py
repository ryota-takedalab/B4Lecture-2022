import argparse
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from io import BytesIO
from PIL import Image

def init_random(data, num_c):
    """define initial centroids randomly
    Args:
        data (ndarray): imput data
        num_c (int): number of clusters
    Returns:
        centroid (ndarray)
    """

    centroid = data[random.sample(range(data.shape[0]), num_c)]

    return centroid

def k_means(data, num_c):
    """clustering by kmeans method
    Args:
        data (ndarray): imput data
        num_c (int): number of clusters
    Returns:
        cluster (ndarray)
        centroid (ndarray) 
    """

    centroid = init_random(data, num_c)  # randomでcentroidの初期値設定
    distance = np.zeros((num_c, data.shape[0]))
    cluster = np.zeros(data.shape[0])

    count = 0
    max_iter = 1000
    
    while(count < max_iter): #各データポイントが属しているclusterが変化しなくなった、または一定回数max_iterの繰り返しを越した時
        #centroid = centroid.copy()
        distance = np.array([np.sum((centroid[i] - data) ** 2, axis = 1) for i in range(num_c)])
        #cluster_bef = cluster
        cluster = np.argmin(distance, axis = 0)  #各列ごとの最大値の行番号
        
        centroid_bef = centroid
        centroid = np.array([np.mean(data[cluster == i], axis = 0) for i in range(num_c)])
        count += 1
        if (centroid_bef == centroid).all():
            print('count:', count)
            break
    
    return cluster, centroid

def render_frame(df, angle, filename):
    """define render frame to make gif
    Args:
        df (ndarray): imput data
        angle (int): angle
        filename (str): filename
    Returns:
        gif images
    """

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df[:, 0], df[:, 1], df[:, 2], c='blueviolet', label='Observed data')
    ax.view_init(30, angle)
    plt.close()

    ax.set(xlabel="$x_1$", ylabel="$x_2$", zlabel="$x_3$", title=f'{filename}')
    ax.legend()

    buf = BytesIO()
    fig.savefig(buf, bbox_inches='tight', pad_inches=0.0)
    return Image.open(buf)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('filepath', type=str, help='csv file name')
    parser.add_argument('c', type=int, help='number of clusters')
    args = parser.parse_args()

    filepath = args.filepath
    filename = filepath.split('.')[0]
    num_c = args.c

    df = pd.read_csv(filepath).values

    cluster, centroid = k_means(df, num_c)
    colors = ['y', 'm', 'c', 'r', 'g', 'b']

    if df.shape[1] == 2:

        fig, ax = plt.subplots(figsize=(7,6))
        ax.set(xlabel="$x_1$", ylabel="$x_2$", title=f'{filename}  c={num_c}')
        ax.scatter(df[:, 0], df[:, 1], c = [colors[i] for i in cluster], label = 'Observed')
        ax.scatter(centroid[:, 0], centroid[:, 1], c='k', label='Centroid')

        # plt.savefig('fig/' + f'{filename}_{num_c}.png')
        plt.show()
        plt.close()

    elif df.shape[1] == 3:

        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set(xlabel="$x_1$", ylabel="$x_2$", zlabel="$x_3$", title=f'{filename}  c={num_c}')
        ax.scatter(df[:, 0], df[:, 1], df[:, 2], c = [colors[i] for i in cluster], label='Observed data')
        ax.scatter(centroid[:, 0], centroid[:, 1], centroid[:, 2], c='k', marker = '*',label='Centroid')
        # plt.savefig('fig/' + f'{filename}_{num_c}.png')
        plt.show()
        plt.close()

        # gif image
        # images = [render_frame(df, angle*2, filename) for angle in range(90)]
        # images[0].save('fig/' + f'{filename}_{num_c}.gif', save_all=True, append_images=images[1:], duration=100, loop=0)

if __name__ == "__main__":
    main()