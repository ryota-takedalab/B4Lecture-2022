import argparse

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from my_functions import k_means
from my_functions import gmm


def main():
    parser = argparse.ArgumentParser(description='ex7')
    parser.add_argument("-i", "--input", help="input file id", type=int)
    parser.add_argument("-n", "--number", help="number of clusters", type=int)
    args = parser.parse_args()
    
    # read data
    data = pd.read_csv(f"../data/data{args.input}.csv").values
    
    # k-means for gmm initialization
    labels, init_mu = k_means.k_means(data, args.number)
    separated_data = k_means.data_separate(data, labels, args.number)
    init_pi = np.array([len(i) for i in separated_data]) / len(data)
    init_sigma = np.array([np.cov(i, rowvar=False) for i in separated_data])
    
    # em algorithm
    gmm_ = gmm.GMM(data.shape[-1], args.number,
                   init_mu=init_mu, init_pi=init_pi, init_sigma=init_sigma)
    gmm_.em(data)
    
    # plot log-likelihood
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    ax[0].set_title(
        f"log-likelihoods: converged in {len(gmm_.log_likelihoods)} iter")
    ax[0].plot(gmm_.log_likelihoods)
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel("log likelihood")
    
    # plot data sample and probability density
    ax[1].set_title(f"EM algorithm on data{args.input}")
    if (data.shape[1] == 1):
        ax[1].scatter(data, np.zeros_like(data), label="data sample",
                      facecolor='None', edgecolor="C0")
        
        # probability density
        ax[1].scatter(gmm_.mu, np.zeros_like(gmm_.mu), label="centroids",
                      marker="x", color="red")
        x1 = np.linspace(np.min(data), np.max(data), 100)
        y1 = np.zeros_like(x1)
        for i in range(gmm_.number):
            y1 += gmm_.gaussian[i].calculate(x1)
        ax[1].plot(x1, y1)
        
    elif (data.shape[1] == 2):
        ax[1].scatter(data[:, 0], data[:, 1], label="data sample",
                      facecolor='None', edgecolor="C0")
        
        # probability density
        ax[1].scatter(gmm_.mu[:, 0], gmm_.mu[:, 1], label="centroids",
                      marker="x", color="red")
        # set sample mesh
        x1 = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 100)
        x2 = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 100)
        xx1, xx2 = np.meshgrid(x1, x2)
        lattice_point_list = np.array([xx1.ravel(), xx2.ravel()]).T
        yy = np.zeros(xx1.size)
        for i in range(gmm_.number):
            yy += gmm_.gaussian[i].calculate(lattice_point_list)
        yy = yy.reshape(xx1.shape)
        mappable = ax[1].contour(xx1, xx2, yy)
        # continuous colorbar
        # https://stackoverflow.com/questions/44498631/continuous-colorbar-with-contour-levels
        norm = matplotlib.colors.Normalize(vmin=mappable.cvalues.min(),
                                           vmax=mappable.cvalues.max())
        sm = plt.cm.ScalarMappable(norm=norm, cmap=mappable.cmap)
        sm.set_array([])
        fig.colorbar(sm, ticks=mappable.levels)
        
    ax[1].legend()
    plt.show()
    
    
if __name__ == "__main__":
    main()
