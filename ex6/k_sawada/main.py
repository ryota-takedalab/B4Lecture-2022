import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

from my_functions import pca


def get_primal_component_edges(data, primal_component):
    """helper for primal component plot
    generate each primal component edge coordinates

    Args:
        data (ndarray, axis=(each data, dimension)): input data
        primal_component (ndarray): primal component

    Returns:
        , axis=(primal component, variable, start and end): edge coordinates
    """
    count, dimension = primal_component.shape
    
    edges = np.zeros((count, dimension, 2))
    start_x = np.min(data[:, 0])
    end_x = np.max(data[:, 0])
    for i in range(count):
        edges[i, 0, 0] = start_x
        edges[i, 0, 1] = end_x
        for j in range(dimension - 1):
            start_y = start_x / primal_component[i][0] * primal_component[i][j + 1]
            end_y = end_x / primal_component[i][0] * primal_component[i][j + 1]
            edges[i, j + 1, 0] = start_y
            edges[i, j + 1, 1] = end_y
    
    return edges


def main():
    # read data with standardization
    data = [
        scipy.stats.zscore(
            pd.read_csv(f"../data{i+1}.csv").values) for i in range(3)
    ]
    
    # plot initialize
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    ax = ax.flatten()
    ax_idx = 0
    
    for i in range(len(data)):
        # pca
        variances, primal_component = pca.pca(data[i])
        
        if (data[i].shape[1] == 2):
            # plot ovserved data
            ax[ax_idx].set_title(f"data{i + 1}")
            ax[ax_idx].set_xlabel("x")
            ax[ax_idx].set_ylabel("y")
            ax[ax_idx].grid()
            ax[ax_idx].scatter(data[i][:, 0], data[i][:, 1], c='black')
            
            # plot primal component
            edges = get_primal_component_edges(data[i], primal_component)
            for pc in range(2):
                ax[ax_idx].plot(edges[pc, 0], edges[pc, 1],
                                label=f"primal component {pc + 1}")
            ax[ax_idx].legend()
            
            ax_idx += 1
            
        elif (data[i].shape[1] == 3):
            ax[ax_idx] = fig.add_subplot(2, 2, ax_idx + 1, projection="3d")
            ax[ax_idx].set_title(f"data{i + 1}")
            ax[ax_idx].set_xlabel("x1")
            ax[ax_idx].set_ylabel("x2")
            ax[ax_idx].set_zlabel("y")
            ax[ax_idx].scatter(data[i][:, 0], data[i][:, 1], data[i][:, 0],
                               c='black')
            ax[ax_idx].set_xlim(np.min(data[i][:, 0]), np.max(data[i][:, 0]))
            ax[ax_idx].set_ylim(np.min(data[i][:, 1]), np.max(data[i][:, 1]))
            ax[ax_idx].set_zlim(np.min(data[i][:, 2]), np.max(data[i][:, 2]))
            # plot primal component
            edges = get_primal_component_edges(data[i], primal_component)
            for pc in range(3):
                ax[ax_idx].plot(edges[pc, 0], edges[pc, 1], edges[pc, 2],
                                label=f"primal component {pc + 1}")
            ax[ax_idx].legend()
            
            ax_idx += 1
            
            # dimension compress to 2d
            compressed, _ = pca.dimension_compress(data[i], primal_component,
                                                   variances, dimension=2)
            ax[ax_idx].set_title(f"data{i + 1} in 2D plane")
            ax[ax_idx].set_xlabel("pc1")
            ax[ax_idx].set_ylabel("pc2")
            ax[ax_idx].grid()
            ax[ax_idx].scatter(compressed[:, 0], compressed[:, 1], c='black')
            
            ax_idx += 1
            
        else:
            # total contribution rate over 90%
            compressed, histories = \
                pca.dimension_compress(data[i], primal_component,
                                       variances, contribution_rate=0.9)
            ax[ax_idx].set_title(
                f"data{i + 1} contribution rate\n"
                f"compression: {data[i].shape[1]} to {compressed.shape[1]}")
            ax[ax_idx].plot(histories)
            ax[ax_idx].plot([0, data[i].shape[1]], [0.9, 0.9], c="r")
            ax[ax_idx].grid()
    plt.show()


if __name__ == "__main__":
    main()
