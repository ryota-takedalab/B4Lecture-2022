import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def gaussian(x, mu, sigma):
    # x.shape = (dim, )
    dim = len(x)
    # sigma.shape = (dim, dim)
    sigma_det = np.linalg.det(sigma)
    sigma_inv = np.linalg.inv(sigma)  # shape = (dim, dim)
    a = ((2 * np.pi) ** (0.5 * dim)) * (sigma_det ** 0.5)
    b = - 0.5 * (x - mu) @ sigma_inv @ (x - mu).T
    gaussian = np.exp(b) / a
    return gaussian


def calc_gaussian(input, mu, sigma):
    # input.shape = (n, dim)
    n = len(input)

    gaussians = np.zeros(n)
    for i in range(n):
        gaussians[i] = gaussian(input[i], mu, sigma)

    return gaussians


def calc_mixture_gaussian(input, pi, mu, sigma):
    n = len(input)
    k = len(pi)

    mix_gaussian = np.zeros((k, n))
    for i in range(k):
        mix_gaussian[i] = pi[i] * calc_gaussian(input, mu[i], sigma[i])

    return mix_gaussian


def calc_loglikelihood(input, pi, mu, sigma):
    n = len(input)
    mix_gaussian = calc_mixture_gaussian(input, pi, mu, sigma)
    sum_mix_gaussian = np.sum(mix_gaussian, axis=0)

    loglikelihood = 0
    for i in range(n):
        loglikelihood += np.log(sum_mix_gaussian[i])

    return loglikelihood


def em_algorithm(input, pi, mu, sigma, epsilon=0.001, max_iter=50):
    n = len(input)
    loglikelihood = []
    loglikelihood.append(calc_loglikelihood(input, pi, mu, sigma))

    for i in range(max_iter):
        # E step
        mix_gaussian = calc_mixture_gaussian(input, pi, mu, sigma)  # shape = (k, n)
        sum_mix_gaussian = np.sum(mix_gaussian, axis=0)  # shape = (n, )
        # (k, n) <- (k, n) / (1, n)
        gamma = mix_gaussian / sum_mix_gaussian[np.newaxis, :]

        # M step
        # update n_k
        n_k = np.sum(gamma, axis=1)  # shape = (k, )

        # update mu
        mu = (gamma @ input) / n_k[:, np.newaxis]  # shape = (k, dim)

        # update sigma
        for k in range(len(n_k)):
            sigma[k] = 0
            for j in range(n):
                sigma[k] += gamma[k][j] * (input[j] - mu[k])[:, np.newaxis] @ (input[j] - mu[k])[np.newaxis, :]
        sigma /= n_k[:, np.newaxis, np.newaxis]  # (k, dim, dim) <- (k, dim, dim) / (k, 1, 1)

        # update pi
        pi = n_k / n

        new_loglikelihood = calc_loglikelihood(input, pi, mu, sigma)
        loglikelihood.append(new_loglikelihood)

        # convergence test
        if loglikelihood[i + 1] - loglikelihood[i] < epsilon:
            break

    return loglikelihood, pi, mu, sigma


def main():
    # read csv files as DataFrame
    df_data1 = pd.read_csv("data1.csv", header=None)
    df_data2 = pd.read_csv("data2.csv", header=None)
    df_data3 = pd.read_csv("data3.csv", header=None)

    # convert DataFrame form to array form
    data1 = df_data1.to_numpy()
    data2 = df_data2.to_numpy()
    data3 = df_data3.to_numpy()

    # set drawing area
    plt.rcParams["figure.figsize"] = (7, 7)

    # fitting data1
    # determine initial parameters for data1
    n = data1.shape[0]
    dim = data1.shape[1]
    k = 2
    pi = np.full(k, 1 / k)
    mu = np.random.randn(k, dim)
    sigma = np.array([np.eye(dim) for _ in range(k)])

    # apply em-algorithm to data1
    loglikelihood1, pi1, mu1, sigma1 = em_algorithm(data1, pi, mu, sigma)

    # set drawing area for data1
    fig1, ax1 = plt.subplots(2, 1)
    # plot log-likelihood
    ax1[0].plot(np.arange(0, len(loglikelihood1), 1), loglikelihood1, c='royalblue')
    ax1[0].set(title="data1 log-likelihood", xlabel="iteration number", ylabel="log-likelihood")
    # plot scatter & mixture gaussian distribution
    x = np.linspace(np.min(data1[:, 0]), np.max(data1[:, 0]), n)
    pdfs = np.zeros((n, k))
    for i in range(k):
        pdfs[:, i] = pi1[i] * norm.pdf(x, loc=mu1[i], scale=sigma1[i])
    ax1[1].scatter(data1, np.zeros(len(data1)), facecolor="None", edgecolors="royalblue", label="data1 sample")
    ax1[1].scatter(mu1, np.zeros(len(mu1)), s=100, marker="*", c="gold", label="centroids")
    ax1[1].plot(x, np.sum(pdfs, axis=1), label="GMM", c='royalblue')
    ax1[1].set(title="GMM data1", xlabel="data1", ylabel="mixture gaussian distribution")
    ax1[1].legend()

    # fitting data2
    # determine initial parameters for data2
    n = data2.shape[0]
    dim = data2.shape[1]
    k = 3
    pi = np.full(k, 1 / k)
    mu = np.random.randn(k, dim)
    sigma = np.array([np.eye(dim) for _ in range(k)])

    # apply em-algorithm to data2
    loglikelihood2, pi2, mu2, sigma2 = em_algorithm(data2, pi, mu, sigma)

    # set drawing area for data2
    fig2, ax2 = plt.subplots(2, 1)
    # plot log-likelihood
    ax2[0].plot(np.arange(0, len(loglikelihood2), 1), loglikelihood2, c='royalblue')
    ax2[0].set(title="data2 log-likelihood", xlabel="iteration number", ylabel="log-likelihood")
    # plot scatter & mixture gaussian distribution
    x = np.linspace(np.min(data2[:, 0]), np.max(data2[:, 0]), n)
    y = np.linspace(np.min(data2[:, 1]), np.max(data2[:, 1]), n)
    X, Y = np.meshgrid(x, y)
    lines = np.dstack((X, Y))
    Z = np.zeros((n, n))
    for i in range(n):
        Z[i] = np.array([np.sum(calc_mixture_gaussian(lines[i, :], pi2, mu2, sigma2), axis=0)])
    ax2[1].scatter(mu2[:, 0], mu2[:, 1], s=100, marker="*", c="r", label="centroid")
    ax2[1].contour(X, Y, Z, cmap="rainbow")
    ax2[1].scatter(df_data2[0], df_data2[1], facecolor="None", edgecolors="royalblue", label="data2 sample")
    ax2[1].set(title="data2")
    ax2[1].legend()

    # fitting data3
    # determine initial parameters for data3
    n = data3.shape[0]
    dim = data3.shape[1]
    k = 3
    pi = np.full(k, 1 / k)
    mu = np.random.randn(k, dim)
    sigma = np.array([np.eye(dim) for _ in range(k)])

    # apply em-algorithm to data3
    loglikelihood3, pi3, mu3, sigma3 = em_algorithm(data2, pi, mu, sigma)

    # set drawing area for data3
    fig3, ax3 = plt.subplots(2, 1)
    # plot log-likelihood
    ax3[0].plot(np.arange(0, len(loglikelihood3), 1), loglikelihood3, c='royalblue')
    ax3[0].set(title="data3 log-likelihood", xlabel="iteration number", ylabel="log-likelihood")
    # plot scatter & mixture gaussian distribution
    x = np.linspace(np.min(data3[:, 0]), np.max(data3[:, 0]), n)
    y = np.linspace(np.min(data3[:, 1]), np.max(data3[:, 1]), n)
    X, Y = np.meshgrid(x, y)
    lines = np.dstack((X, Y))
    Z = np.zeros((n, n))
    for i in range(n):
        Z[i] = np.array([np.sum(calc_mixture_gaussian(lines[i, :], pi3, mu3, sigma3), axis=0)])
    ax3[1].scatter(mu3[:, 0], mu3[:, 1], s=100, marker="*", c="r", label="centroid")
    ax3[1].contour(X, Y, Z, cmap="rainbow")
    ax3[1].scatter(df_data3[0], df_data3[1], facecolor="None", edgecolors="royalblue", label="data3 sample")
    ax3[1].set(title="data3")
    ax3[1].legend()

    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    fig1.savefig("data1.png")
    fig2.savefig("data2.png")
    fig3.savefig("data3.png")


if __name__ == "__main__":
    main()
