import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class GMM:
    def __init__(self, data, method):

        """
        self.data : input data
        self.mu, self.sig, self.pi_k : 混合ガウス分布のパラメータの行列
        """

        self.data = data
        self.num = data.shape[0]
        self.deg = data.shape[1]
        self.k = 3
        self.e = 0.001

        if self.deg == 1:
            if method == "km":
                model = KMeans(n_clusters=self.k, init="random")
                model.fit(data)
                mu = model.cluster_centers_
        
            np.random.seed(1)
            sig = np.random.rand(self.k, 1, 1)

        elif self.deg == 2:
            if method == "km":
                model = KMeans(n_clusters=self.k, init="random")
                model.fit(data)
                mu = model.cluster_centers_
            
            np.random.seed(1)
            sig = 10 * np.random.rand(self.k, 2, 2) - 5


        pi = np.zeros(self.k)
        pi += 1 / self.k

        self.mu = mu
        self.sigma = sig
        self.pi = pi


    def gaus1(self):
        data = self.data
        sigma = self.sigma
        mu = self.mu
        gaus = np.exp(- pow(( data - mu ), 2) / (2 * sigma)) / np.sqrt(2 * np.pi * sigma)

        return gaus


    def gaus(self, data):
        # 正規分布の値の取得
        # (N, D) - (k, D) ->(1, N, D) - (k, 1, D) = (k, N, D)
        self.diff_data = data[np.newaxis, :, :] - self.mu[:, np.newaxis, :]
        # (k, N, N)
        tmp = (
            self.diff_data
            @ np.linalg.inv(self.sigma)
            @ self.diff_data.transpose(0, 2, 1)
        )
        # 分子
        nume = np.exp(-np.diagonal(tmp, axis1=1, axis2=2,) / 2.0)
        # 分母
        deno = np.sqrt(
            ((2 * np.pi) ** self.deg) * np.abs(np.linalg.det(self.sigma))
        ).reshape(-1, 1)

        gaus = nume / deno
        return gaus


    def calc_gamma(self):
        # 負担率の計算
        self.gamma = self.pi.reshape(-1, 1) * self.gaus(self.data)
        self.gamma /= np.sum(self.gamma, axis=0)


    def update_params(self):
        # pi, mu, sigmaの更新
        self.N_k = np.sum(self.gamma, axis=1)
        self.pi = self.N_k / np.sum(self.N_k)
        self.mu = np.sum(
            self.data * self.gamma[:, :, np.newaxis], axis=1) / self.N_k.reshape(-1, 1)
        self.sigma = (
            self.gamma[:, np.newaxis, :]
            * self.diff_data.transpose(0, 2, 1)
            @ self.diff_data
        ) / self.N_k.reshape(-1, 1, 1)
        

    def EM(self):
        LF = 0
        LF_list = []
        while True:
            self.calc_gamma()
            #対数尤度関数
            LF_new = np.sum(np.log(np.sum(self.pi.reshape(-1, 1) * self.gaus(self.data), axis=0)))
            LF_list.append(LF_new)
            ch = LF_new - LF
            if np.abs(ch) < self.e:
                self.LF_list = LF_list
                break
            LF = LF_new

            self.update_params()


    def p(self, xline) :
        x = self.data
        px = np.zeros_like(xline)
        mu = self.mu
        sigma = self.sigma
        pi = self.pi
        px = pi @ self.gaus(xline)
        return px


    def scatter(self, ax):
        x = self.data[:, 0]
        if self.deg == 1:
            y = np.zeros_like(self.data)
        elif self.deg == 2:
            y = self.data[:, 1]
        ax.scatter(x, y, label="Data Sample", facecolor="None", edgecolor="blue")


    def plot_mu(self, ax):
        x = self.mu[:, 0]
        if self.deg == 1:
            y = np.zeros_like(self.mu)
        else:
            y = self.mu[:, 1]
        ax.scatter(x, y, label="Centroids", marker="x", color="red")
        

    def plot_gmm(self, ax):
        self.plot_mu(ax)
        if self.deg == 1:
            x = self.data
            xline = np.array([np.linspace(x.min(), x.max())]).T
            y = self.p(xline)
            ax.plot(xline, y, label="GMM")

        elif self.deg == 2:
            x = self.data[:, 0]
            y = self.data[:, 1]
            xx = np.linspace(x.min(), x.max())
            yy = np.linspace(y.min(), y.max())
            X, Y = np.meshgrid(xx, yy)
            lines = np.dstack((X, Y))
            pos = 0
            for k in range(self.k):
                pos += np.array([(self.pi.reshape(-1, 1) * self.gaus(line))[k] for line in lines])
            plt.contour(X, Y, pos, cmap="rainbow")
    

    def plot_iter(self, ax):
        lf = self.LF_list
        xline = np.arange(len(lf))
        ax.plot(xline[1:], lf[1:])


if __name__ == "__main__":

    data1 = pd.read_csv('../data/data1.csv').values
    data2 = pd.read_csv('../data/data2.csv').values
    data3 = pd.read_csv('../data/data3.csv').values


    model1 = GMM(data1, "km")
    model1.EM()
    fig = plt.figure()
    ax = fig.add_subplot()
    model1.scatter(ax)
    model1.plot_gmm(ax)
    ax.set(xlabel="x", ylabel="Probability density", title="data1 GMM")
    ax.legend()
    plt.show()
    plt.savefig("data1.png")

    fig = plt.figure()
    ax = fig.add_subplot()
    model1.plot_iter(ax)
    ax.set(xlabel="iteration", ylabel="Log Likelihood", title="data1 Lilelihood")
    plt.show()
    plt.savefig("data1_LF.png")
    

    model2 = GMM(data2, "km")
    model2.EM()

    fig = plt.figure()
    ax = fig.add_subplot()
    model2.scatter(ax)
    model2.plot_gmm(ax)
    ax.set(xlabel="x", ylabel="y", title="data2 GMM")
    ax.legend()
    plt.show()
    plt.savefig("data2.png")
    
    fig = plt.figure()
    ax = fig.add_subplot()
    model2.plot_iter(ax)
    ax.set(xlabel="iteration", ylabel="Log Likelihood", title="data2 Lilelihood")
    plt.show()
    plt.savefig("data2_LF.png")
    

    model3 = GMM(data3, "km")
    model3.EM()
    fig = plt.figure()
    ax = fig.add_subplot()
    model3.scatter(ax)
    model3.plot_gmm(ax)
    ax.set(xlabel="x", ylabel="y", title="data3 GMM")
    ax.legend()
    plt.show()
    plt.savefig("data3.png")
    
    fig = plt.figure()
    ax = fig.add_subplot()
    model2.plot_iter(ax)
    ax.set(xlabel="iteration", ylabel="Log Likelihood", title="data3 Lilelihood")
    plt.show()
    plt.savefig("data3_LF.png")
     