import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class GMM:
    def __init__(self, data):

        """
        self.data : input data
        self.mu, self.sig, self.pi_k : 混合ガウス分布のパラメータの行列

        """

        self.x = data
        self.mu = np.array([-3, 11])
        self.sigma = np.array([4, 5])
        self.pi = np.array([0.1, 0.9])
        self.e = 0.01


    def gauss(self, x, ave, var):

        gauss = np.exp(- pow(( x - ave ), 2) / (2 * var)) / np.sqrt(2 * np.pi * var)
        return gauss


    def calc_gamma(self):
        self.gamma = self.pi * self.gauss(self.x, self.mu, self.sigma)
        self.gamma /= np.sum(self.gamma, axis=1).reshape(len(self.x), 1)


    def update_params(self):
        self.N = np.sum(self.gamma, axis=0)
        self.mu = np.sum(self.x * self.gamma, axis=0) / self.N
        self.sigma = np.sum(self.gamma * pow(self.x - self.mu, 2), axis=0) / self.N
        self.pi = self.N / np.sum(self.N)


    def EM(self):
        LF = 0
        while True:
            self.calc_gamma()
            #対数尤度関数
            LF_new = np.sum(np.log(np.sum(self.pi * self.gauss(self.x, self.mu, self.sigma), axis=1)))
            ch = LF_new - LF
            if np.abs(ch) < self.e:
                print(self.mu)
                print(self.sigma)
                print(self.pi)
                break
            LF = LF_new
            self.update_params()


    def p(self, xline) :
        x = self.x
        px = np.zeros_like(xline)
        mu = self.mu
        sigma = self.sigma
        pi = self.pi
        for i in range(self.mu.shape[0]):
            px += pi[i] * np.exp(- pow(( xline - mu[i] ), 2) / (2 * sigma[i])) / np.sqrt(2 * np.pi * sigma[i])
        return px


    def scatter1d(self, ax):
        data_size = self.x.shape[1]
        x = self.x
        y = np.zeros_like(x)
        ax.scatter(x, y, label="Data Sample")


    def plot_gmm(self,ax):
        y = np.zeros_like(self.mu)
        ax.scatter(self.mu, y, label="Centroids")
        x = self.x
        xline = np.linspace(x.min(), x.max())
        y = self.p(xline)
        ax.plot(xline, y, label="GMM")



if __name__ == "__main__":

    data1 = pd.read_csv('../data/data1.csv').values
    data2 = pd.read_csv('../data/data2.csv').values
    data3 = pd.read_csv('../data/data3.csv').values

    model1 = GMM(data1)
    model1.EM()
    fig = plt.figure()
    ax = fig.add_subplot()

    model1.scatter1d(ax)
    model1.plot_gmm(ax)

    ax.set(xlabel="x", ylabel="Probability density", title="K=2")
    ax.legend()
    plt.show()
    plt.savefig("data1.png")
    #model2 = GMM(data2)
    #model3 = GMM(data3)
