import argparse
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class GMM:
    def __init__(self, data, k):
        """
        Parameters
        ----------
        self.data [np.ndarray]: Audio data
        k [int]: Number of Cluster
        self.Mu [int]: The random initial centroid value
        self.Sigma [np.ndarray]: The initial value of covariance matrix
        self.Pi [np.ndarray]: The initial value of the mixing coefficient
        """
        self.data = data
        self.n_sample = data.shape[0]
        self.dim = data.shape[1]
        self.Mu = np.random.randn(k, self.dim)
        self.Sigma = np.array([np.eye(self.dim) for i in range(k)])
        self.Pi = np.array([1 / k for i in range(k)])
        self.k = k

    def gauss(self):
        """
        Calculate probability of the k-th gaussian samples
        Returns
        -------
        gauss [np.ndarray]:
        """
        diff_data = self.data - self.Mu[:, None]
        dis = diff_data @ np.linalg.inv(self.Sigma) @ diff_data.transpose(0, 2, 1)
        dis = np.diagonal(dis, axis1=1, axis2=2)  # (k, N)
        num = np.exp(-dis / 2)
        den = np.sqrt((2 * np.pi) ** self.dim) * np.sqrt(np.linalg.det(self.Sigma))
        gauss = num/den[:, None]
        return gauss

    def gmm(self):
        """
        Calculate probability that each sample belongs to each class
        Returns
        -------
        prob [np.ndarray]: GMM Probability Density
        w_gauss [np.ndarray]: Used for covariance
        """
        w_gauss = self.gauss() * self.Pi[:, None]
        prob = np.sum(w_gauss, axis=0)
        return prob, w_gauss

    def em(self, e=1e-16):
        """
        EM Algorithm
        Args
        -------
            e [float] = error limit
        Returns
        -------
            log_list [list]: log-likelihood
            self.Pi [ndarray]: New mixing coefficient
            self.Mu [ndarray]: New centroid
            self.Sigma [ndarray]: New covariance matrix
        """
        prob, w_gauss = self.gmm()
        log_list = [np.sum(np.log(prob))]
        n = 0
        max_iter = 100
        while True:
            # E step
            gamma = w_gauss/prob
            # M step
            # update n_k, mu, and pi
            n_k = gamma.sum(axis=1)
            self.Mu = np.sum(gamma[:, :, None] * self.data / n_k[:, None, None], axis=1)
            diff_x = self.data - self.Mu[:, None]  # (k, N, D)
            self.Sigma = (gamma[:, None] * diff_x.transpose(0, 2, 1) @ diff_x) / n_k[:, None, None]
            self.Pi = n_k / self.n_sample
            prob, w_gauss = self.gmm()
            # update log-likelihood function
            log_likelihood = np.sum(np.log(prob))
            log_list.append(log_likelihood)
            n += 1
            if log_likelihood - log_list[-2] < e or n == max_iter:
                return self.Mu, self.Sigma, self.Pi, log_list, n


def main():
    # Argparse
    parser = argparse.ArgumentParser(description='Name of the Data File')
    parser.add_argument('-fn', metavar='-f', dest='filename', type=str,
                        help='Enter the Data Name', required=True)
    parser.add_argument('-k', dest='ncluster', type=str,
                        help='Enter the Number of Cluster', required=True)
    args = parser.parse_args()

    # Read data
    name = args.filename
    k = args.ncluster
    data = pd.read_csv(f'data/{name}.csv')
    data_val = data.values

    # Log-likelihood
    gmm = GMM(data_val, k)
    gmm.Mu, gmm.Sigma, gmm.Pi, log_list, n = gmm.em(e=1e-4)
    plt.plot(log_list)
    plt.title(f'Log-Likelihood\nk={k}, {name}, iter={n}')
    plt.xlabel('Iteration')
    plt.ylabel('Log-likelihood')
    plt.tight_layout()
    plt.savefig(f'figs/log_likelihood_{k}_{name}.png')
    plt.show()

    if len(data.columns) == 1:
        plt.figure(figsize=(5, 5))
        ax = plt.subplot()
        ax.scatter(data_val, np.zeros(data_val.shape[0]), edgecolors='green', facecolor='None',
                   label='Data')
        ax.plot(gmm.Mu, np.zeros(gmm.Mu.shape[0]), marker='x',
                label='Centroid', color='red', markersize='10', linestyle='')
        x1 = np.linspace(np.min(data_val) - 1, np.max(data_val) + 1, 100)[:, None]
        gmm.data = x1
        prob, w_gauss = gmm.gmm()
        ax.plot(x1, prob, label="GMM")
        ax.set_title(f'K={gmm.k}, {name}')
        ax.set_xlabel('x1')
        ax.set_ylabel('probability')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'figs/probability_k_{gmm.k}_{name}.png')
        plt.show()

    elif len(data.columns) == 2:
        x1 = np.linspace(np.min(data_val[:, 0]) - 1, np.max(data_val[:, 0]) + 1, 100)
        x2 = np.linspace(np.min(data_val[:, 1]) - 1, np.max(data_val[:, 1]) + 1, 100)
        # Make a (100,100)(100,100) mesh grid
        x1, x2 = np.meshgrid(x1, x2)
        lines = np.dstack((x1, x2))
        probability = np.array([gmm.gmm()[0] for gmm.data in lines])
        # plot contour map
        plt.figure(figsize=(6, 6))
        ax = plt.subplot()
        ax.scatter(data_val[:, 0], data_val[:, 1], c="white", linewidths=1,
                   edgecolors="red", label='data')
        ax.scatter(gmm.Mu[:, 0], gmm.Mu[:, 1], marker='*', c="white", linewidths=2,
                   edgecolors="blue", label='Centroid', s=100)
        contour = ax.contour(x1, x2, probability, cmap='jet')
        ax.set_title(f'K={k}, {name}')
        # From https://matplotlib.org/2.0.2/examples/pylab_examples/contour_demo.html
        norm = matplotlib.colors.Normalize(vmin=contour.cvalues.min(), vmax=contour.cvalues.max())
        sm = plt.cm.ScalarMappable(norm=norm, cmap=contour.cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ticks=contour.levels)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'figs/contour_probability_k_{k}_{name}.png')
        plt.show()


if __name__ == '__main__':
    main()
