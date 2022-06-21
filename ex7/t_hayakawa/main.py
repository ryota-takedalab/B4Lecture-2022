import numpy as np
import argparse
import time
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.cluster import KMeans


class GMM:
    def __init__(self, N, D, K, mu, sigma, pi) -> None:
        """

        Args:
            N(int): data size
            D (int): dimension
            K (int): cluster num
            mu (ndarray, shape=(K, D)): mean value
            sigma (ndarray, shape=(K, D, D)): covariance
            pi (ndarray, shape=(K, )): mixing coefficient
        """
        # self.data = data
        self.N = N
        self.D = D
        self.K = K
        self.mu = mu
        self.sigma = sigma
        self.pi = pi

    def gaus(self, data):
        """calculate gaussian value

        Args:
            data (ndarray, shape=(N, D)): input data

        Returns:
            ndarray, shape=(K, N): gaussian value
        """
        # 正規分布の値の取得
        # (N, D) - (k, D) ->(1, N, D) - (k, 1, D) = (k, N, D)
        diff_data = data[np.newaxis, :, :] - self.mu[:, np.newaxis, :]
        # (k, N, N)
        tmp = diff_data @ np.linalg.inv(self.sigma) @ diff_data.transpose(0, 2, 1)
        # 分子
        nume = np.exp(-np.diagonal(tmp, axis1=1, axis2=2,) / 2.0)
        # 分母
        deno = np.sqrt(
            ((2 * np.pi) ** self.D) * np.abs(np.linalg.det(self.sigma))
        ).reshape(-1, 1)

        return nume / deno

    def iteration(self, data, I=100, e=0.01):
        """calculate likelihood until ( change < e )

        Args:
            data (ndarray, shape=(N, D)): input data
            I (int, optional): max iteration times. Defaults to 100.
            e (float, optional): threshold. Defaults to 0.01.
        
        Returns:
            ndarray(K, ) : pi
            ndarray(K, D) : mu
            ndarray(K, D, D) : sigma
            ndarray(iteration, 2) : likelihood list
            int : iteration
        """

        # ε以下になるか、100回計算するまで尤度を更新
        lh = 0
        lh_list = np.empty((0, 2))
        for i in range(I):
            # 負担率ガンマの計算
            gam = self.pi.reshape(-1, 1) * self.gaus(data)
            gam /= np.sum(gam, axis=0)
            self.gamma = gam

            # 対数尤度を計算
            lh_new = np.sum(
                np.log(np.sum(self.pi.reshape(-1, 1) * self.gaus(data), axis=0))
            )
            lh_list = np.vstack([lh_list, [i, lh_new]])
            # 対数尤度の変化量
            ch = lh_new - lh
            if np.abs(ch) < e:
                print(f"Iteration is finished {i+1} iter.")
                break
            lh = lh_new

            # pi, mu, sigmaの更新
            N_k = np.sum(self.gamma, axis=1)

            self.pi = N_k / self.N

            self.mu = np.sum(self.gamma[:, :, np.newaxis] * data, axis=1) / N_k.reshape(
                -1, 1
            )

            diff_data = data[np.newaxis, :, :] - self.mu[:, np.newaxis, :]
            self.sigma = (
                self.gamma[:, np.newaxis, :] * diff_data.transpose(0, 2, 1) @ diff_data
            ) / N_k.reshape(-1, 1, 1)

        return self.pi, self.mu, self.sigma, lh_list, i + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="data file .csv")
    parser.add_argument("--k", type=int, required=True, help="cluster number")
    parser.add_argument(
        "--m",
        type=str,
        choices=["r", "km", "km+"],
        default="r",
        help="method for deciding first mu, [random, k-means, k-means++]",
    )

    args = parser.parse_args()

    # load text from .csv
    data = pd.read_csv(args.filename + ".csv", header=None).values

    K = args.k
    method = args.m
    n = data.shape[0]

    if data.shape[1] == 1:
        start = time.time()
        if method == "r":
            # 平均値
            mu = (np.max(data[:, 0]) - np.min(data[:, 0])) * np.random.rand(
                K, 1
            ) + np.min(data[:, 0])

            # 0~1のランダムで分散
            sig = np.random.rand(K, 1, 1)

            # 混合係数
            pi = np.zeros(K)
            pi += 1 / K
        elif method == "km":
            model = KMeans(n_clusters=K, init="random")
            fit_data = model.fit_predict(data)

            # 平均値
            mu = model.cluster_centers_

            # 分散
            index_data = [np.where(fit_data == i) for i in range(K)]
            sig = [[[np.var(data[index_data[i]])]] for i in range(K)]

            # 混合係数
            pi = np.zeros(K)
            for k in range(K):
                pi[k] = index_data[k][0].shape[0] / data.shape[0]
        elif method == "km+":
            model = KMeans(n_clusters=K)
            fit_data = model.fit_predict(data)

            # 平均値
            mu = model.cluster_centers_

            # 分散
            index_data = [np.where(fit_data == i) for i in range(K)]
            sig = [[[np.var(data[index_data[i]])]] for i in range(K)]

            # 混合係数
            pi = np.zeros(K)
            for k in range(K):
                pi[k] = index_data[k][0].shape[0] / data.shape[0]

        gmm = GMM(data.shape[0], data.shape[1], K, mu, sig, pi)
        pi, mu, sig, lh_list, iter = gmm.iteration(data, I=100, e=0.01)

        calc_time = time.time() - start
        print(f"calc_time({method}): {calc_time:.5f}")

        xx = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), n)
        pdfs = np.zeros((n, K))
        for k in range(K):
            pdfs[:, k] = pi[k] * norm.pdf(xx, loc=mu[k], scale=sig[k])

        fig = plt.figure(figsize=(8, 6))
        plt.subplots_adjust(hspace=0.6)
        fig.add_subplot(
            211,
            title=f"{args.filename} GMM ({iter} iter)",
            xlabel="x",
            ylabel="probability",
        )
        plt.scatter(
            data,
            np.zeros(data.shape[0]),
            facecolor="None",
            edgecolors="darkblue",
            label="data",
        )
        plt.scatter(
            mu, np.zeros(mu.shape[0]), s=100, marker="x", c="r", label="centroid",
        )
        plt.plot(xx, np.sum(pdfs, axis=1), label="GMM")
        plt.legend()

        fig.add_subplot(
            212,
            title=f"{args.filename} GMM stack ({iter} iter)",
            xlabel="x",
            ylabel="probability",
        )
        labels = []
        stacks = np.zeros((0, pdfs.shape[0]))
        for k in range(K):
            labels.append(f"$\pi{k}$ : {pi[k]:.3f}")
            stacks = np.vstack((stacks, pdfs[:, k]))
        plt.stackplot(xx, stacks, labels=labels)
        plt.legend(bbox_to_anchor=(0.25, 1))
        plt.savefig(f"{args.filename}_gmm.png")
        plt.show()

        # 対数尤度関数
        fig = plt.figure()
        fig.add_subplot(
            111,
            title=f"{args.filename} Likelihood",
            xlabel="Iteration",
            ylabel="Log Likelihood",
        )
        plt.plot(lh_list[:, 0], lh_list[:, 1])
        plt.savefig(f"{args.filename}_lh.png")
        plt.show()

    elif data.shape[1] == 2:
        start = time.time()

        if method == "r":
            mu_x = (np.max(data[:, 0]) - np.min(data[:, 0])) * np.random.rand(
                K, 1
            ) + np.min(data[:, 0])
            mu_y = (np.max(data[:, 1]) - np.min(data[:, 1])) * np.random.rand(
                K, 1
            ) + np.min(data[:, 1])
            mu = np.hstack((mu_x, mu_y))

            # 単位行列 * 正の乱数
            sig = np.random.rand(K, 2, 2) * np.identity(2)

            pi = np.zeros(K)
            pi += 1 / K
        elif method == "km":
            model = KMeans(n_clusters=K, init="random")
            fit_data = model.fit_predict(data)
            mu = model.cluster_centers_

            # 共分散
            index_data = [np.where(fit_data == i) for i in range(K)]
            sig = [[np.cov(data[index_data[i]].T)] for i in range(K)]
            sig = np.squeeze(sig)

            # 混合係数
            pi = np.zeros(K)
            for k in range(K):
                pi[k] = index_data[k][0].shape[0] / data.shape[0]
        elif method == "km+":
            model = KMeans(n_clusters=K)
            fit_data = model.fit_predict(data)
            mu = model.cluster_centers_

            # 共分散
            index_data = [np.where(fit_data == i) for i in range(K)]
            sig = [[np.cov(data[index_data[i]].T)] for i in range(K)]
            sig = np.squeeze(sig)

            # 混合係数
            pi = np.zeros(K)
            for k in range(K):
                pi[k] = index_data[k][0].shape[0] / data.shape[0]

        gmm = GMM(data.shape[0], data.shape[1], K, mu, sig, pi)
        pi, mu, sig, lh_list, iter = gmm.iteration(data, I=100, e=0.01)

        calc_time = time.time() - start
        print(f"calc_time({method}): {calc_time:.5f}")

        fig = plt.figure()
        fig.add_subplot(
            111, title=f"{args.filename} GMM ({iter} iter)", xlabel="x", ylabel="y",
        )
        plt.scatter(
            data[:, 0],
            data[:, 1],
            facecolor="None",
            edgecolors="darkblue",
            label="data",
        )
        plt.scatter(
            mu[:, 0], mu[:, 1], s=100, marker="x", c="r", label="centroid",
        )

        xx = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), n)
        yy = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), n)

        X, Y = np.meshgrid(xx, yy)
        lines = np.dstack((X, Y))
        pos = 0
        for k in range(K):
            pos += np.array([pi[k] * gmm.gaus(line)[k] for line in lines])

        plt.contour(X, Y, pos, cmap="rainbow")
        plt.legend()
        plt.savefig(f"{args.filename}_gmm.png")
        plt.show()

        # 対数尤度関数
        fig = plt.figure()
        fig.add_subplot(
            111,
            title=f"{args.filename} Likelihood",
            xlabel="Iteration",
            ylabel="Log Likelihood",
        )
        plt.plot(lh_list[:, 0], lh_list[:, 1])
        plt.savefig(f"{args.filename}_lh.png")
        plt.show()


if __name__ == "__main__":
    main()

