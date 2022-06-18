import numpy as np
import argparse
import time
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.cluster import KMeans


class GMM:
    def __init__(self, data, K, mu, sigma, pi) -> None:
        self.data = data
        self.N = data.shape[0]
        self.D = data.shape[1]
        self.K = K
        self.mu = mu
        self.sigma = sigma
        self.pi = pi

    def gaus(self):
        # 正規分布の値の取得
        # (N, D) - (k, D) ->(1, N, D) - (k, 1, D) = (k, N, D)
        self.diff_data = self.data[np.newaxis, :, :] - self.mu[:, np.newaxis, :]
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
            ((2 * np.pi) ** self.D) * np.abs(np.linalg.det(self.sigma))
        ).reshape(-1, 1)

        return nume / deno

    def calc_gamma(self):
        # 負担率の計算
        gam = self.pi.reshape(-1, 1) * self.gaus()
        gam /= np.sum(gam, axis=0)
        self.gamma = gam

    def update_parames(self):
        # pi, mu, sigmaの更新
        N_k = np.sum(self.gamma, axis=1)
        self.pi = N_k / self.N
        self.mu = np.sum(
            self.gamma[:, :, np.newaxis] * self.data, axis=1
        ) / N_k.reshape(-1, 1)
        self.sigma = (
            self.gamma[:, np.newaxis, :]
            * self.diff_data.transpose(0, 2, 1)
            @ self.diff_data
        ) / N_k.reshape(-1, 1, 1)

    def iteration(self, I=100, e=0.01):
        # ε以下になるか、100回計算するまで尤度を更新
        lh = 0
        lh_list = np.empty((0, 2))
        for i in range(I):
            # 負担率ガンマの計算
            self.calc_gamma()
            # 対数尤度を計算
            lh_new = np.sum(
                np.log(np.sum(self.pi.reshape(-1, 1) * self.gaus(), axis=0))
            )
            lh_list = np.vstack([lh_list, [i, lh_new]])
            # 対数尤度の変化量
            ch = lh_new - lh
            if np.abs(ch) < e:
                print(f"Iteration is finished {i+1} iter.")
                break
            lh = lh_new
            self.update_parames()
        return self.pi, self.mu, self.sigma, lh_list, i + 1


def gaus_pi(data, pi, mu, sigma, D):
    # pi * 正規分布の値の取得
    # (N, D) - (k, D) ->(1, N, D) - (k, 1, D) = (k, N, D)
    diff_data = data[np.newaxis, :, :] - mu[:, np.newaxis, :]
    # (k, N, N)
    tmp = diff_data @ np.linalg.inv(sigma) @ diff_data.transpose(0, 2, 1)
    # 分子
    nume = np.exp(-np.diagonal(tmp, axis1=1, axis2=2,) / 2.0)
    # 分母
    deno = np.sqrt(((2 * np.pi) ** D) * np.abs(np.linalg.det(sigma))).reshape(-1, 1)
    return (nume / deno) * pi.reshape(-1, 1)


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
            mu = (np.max(data[:, 0]) - np.min(data[:, 0])) * np.random.rand(
                K, 1
            ) + np.min(data[:, 0])
        elif method == "km":
            model = KMeans(n_clusters=K, init="random")
            model.fit(data)
            mu = model.cluster_centers_
        elif method == "km+":
            model = KMeans(n_clusters=K)
            model.fit(data)
            mu = model.cluster_centers_
        # 0~1のランダムで分散
        np.random.seed(1)
        sig = np.random.rand(K, 1, 1)
        pi = np.zeros(K)
        pi += 1 / K
        gmm = GMM(data, K, mu, sig, pi)
        pi, mu, sig, lh_list, iter = gmm.iteration(I=100, e=0.01)

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
        plt.plot(lh_list[1:, 0], lh_list[1:, 1])
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
        elif method == "km":
            model = KMeans(n_clusters=K, init="random")
            model.fit(data)
            mu = model.cluster_centers_
        elif method == "km+":
            model = KMeans(n_clusters=K)
            model.fit(data)
            mu = model.cluster_centers_

        # -5~5の間でランダムな値の共分散
        np.random.seed(1)
        sig = 10 * np.random.rand(K, 2, 2) - 5
        pi = np.zeros(K)
        pi += 1 / K
        gmm = GMM(data, K, mu, sig, pi)
        pi, mu, sig, lh_list, iter = gmm.iteration(I=100, e=0.01)

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
            pos += np.array([gaus_pi(line, pi, mu, sig, 2)[k] for line in lines])

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
        plt.plot(lh_list[1:, 0], lh_list[1:, 1])
        plt.savefig(f"{args.filename}_lh.png")
        plt.show()


if __name__ == "__main__":
    main()
