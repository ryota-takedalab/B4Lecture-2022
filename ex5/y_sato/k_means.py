import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import axes3d

class KMeans:
    def __init__(self, data):
        self.data = data
        self.cluster_num = 3
        self.length = data.shape[0]
        self.dimension = data.shape[1]
        self.belongs = np.zeros(self.length)


    def ini_centroids(self):
        centroids = np.random.randint(0, self.length, size=self.cluster_num)
        centroids = self.data[centroids]
        return centroids


    def fit(self, cluster_num=3):
        self.cluster_num = cluster_num
        data = self.data
        length = self.length
        dimension = self.dimension

        centroids = self.ini_centroids()
        distances = np.zeros([length, cluster_num])
        while True:
            for i in range(length):
                distances[i] = np.linalg.norm(data[i] - centroids, axis=1)

            belongs = np.argmin(distances, axis=1)

            #update centroids
            centroids_new = np.zeros([cluster_num, dimension])
            for c in range(cluster_num):
                centroids_new[c] = np.average(data[belongs == c], axis=0)

            if (np.all(centroids_new == centroids)):
                break

            centroids = centroids_new
            self.belongs = belongs

        return belongs


    def classify(self):
        data = self.data
        belongs = self.belongs
        data_size = data.shape[1]

        fig = plt.figure()
        for cluster_num in range(2, 6):
            self.fit(cluster_num)
            belongs = self.belongs

            #plot
            if data_size == 2 :
                ax = fig.add_subplot(2, 2, self.cluster_num - 1)
                for c in range(self.cluster_num):
                    x = data[belongs == c][:, 0]
                    y = data[belongs == c][:, 1]
                    ax.scatter(x, y)
                    ax.set(xlabel="$x_1$", ylabel="$x_2$")

            else:
                #projectionを後から変更できないのでここの一括化は無理
                ax = fig.add_subplot(2, 2, self.cluster_num - 1, projection="3d")
                for c in range(self.cluster_num):
                    x = data[belongs == c][:, 0]
                    y = data[belongs == c][:, 1]
                    z = data[belongs == c][:, 2]
                    ax.scatter3D(x, y, z)
                    ax.set(xlabel="$x_1$", ylabel="$x_2$", zlabel="$x_3$")

        plt.show()





if __name__ == "__main__":

    data1 = pd.read_csv('data1.csv').values
    data2 = pd.read_csv('data2.csv').values
    data3 = pd.read_csv('data3.csv').values

    model1 = KMeans(data1)
    model1.classify()

    model2 = KMeans(data2)
    model2.classify()

    model3 = KMeans(data3)
    model3.classify()
