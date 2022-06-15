import argparse
import os

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import decimal

def open_csv(fname):
    """
    open csv file

    Parameter
    ---------
    fname : str
        file name
        
    Return
    ------
    data : numpy.array
        opened file in the form of numpy
    """
    data = pd.read_csv(fname, header=None)
    data = np.array(data)

    return data


def standardize_data(data):
    """
    standardize data

    Parameter
    ---------
    data : numpy.ndarray
        observed data

    Return
    ------
    s_data : numpy.ndarray
        standardized data
    """
    # calculate mean of data
    m_data = np.mean(data, axis=0)
    # calculate variance of data
    var_data = np.std(data, axis=0, ddof=1)
    s_data = (data-m_data) / var_data

    return s_data


def pca(data):
    """
    calculate pca

    Parameter
    ---------
    data : numpy.ndarray
        standardized data

    Returns
    -------
    sort_ev : numpy.ndarray
        sorted eigenvector
    contribution : np.ndarray
        contribution rate
    """
    # calculate covariance matrix
    cov = np.cov(data, rowvar=False)
    # calculate eigenvalues and eigenvectors
    lam, ev = np.linalg.eigh(cov)
    # sort eigenvalues and aigenvectors in descending order
    sort_lam = np.sort(lam)[::-1]
    sort_ev = ev[:, np.argsort(lam)[::-1]]
    contribution = sort_lam / np.sum(sort_lam)

    return sort_ev, contribution


def plot_2d(data, ev, contribution, graph_title):
    """
    plot 2d data

    Parameters
    ----------
    data : numpy.ndarray
        observed data
    ev : numpy.ndarray
        eigenvector
    contribution : np.ndarray
        contribution rate
    graph_title : str
        graph title
    """
    x = data[:, 0]
    y = data[:, 1]
    x_label = np.linspace(min(data[:, 0]), max(data[:, 0]), 100)
    z1 = (ev[1, 0] / ev[0, 0]) * x_label
    z2 = (ev[1, 1] / ev[0, 1]) * x_label

    fig, ax = plt.subplots(figsize=(6,8), tight_layout=True,)
    ax.scatter(x, y, c="m", label="data")
    ax.plot(x_label, z1, c="b", label=f"Contribution rate: {round(contribution[0], 3)}")
    ax.plot(x_label, z2, c="g", label=f"Contribution rate: {round(contribution[1], 3)}")
    ax.set_aspect("equal")
    plt.title(graph_title, fontsize=25)
    plt.xlabel("X1", fontsize=20)
    plt.ylabel("X2", fontsize=20)
    plt.grid()
    plt.legend()
    path = os.path.dirname(__file__)
    save_fname = os.path.join(path, "result", graph_title+"_2d.png")
    plt.savefig(save_fname)
    plt.tight_layout()
    
    plt.show()
    plt.close()


def plot_3d(data, ev, contribution, graph_title):
    """
    plot 3d data

    Parameters
    ----------
    data : numpy.ndarray
        observed data
    ev : numpy.ndarray
        eigenvector
    contribution : np.ndarray
        contribution rate
    graph_title : str
        graph title
    """
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    x_label = np.linspace(min(data[:, 0]), max(data[:, 0]), 100)
    x_y = ev[1, :] / ev[0, :]
    x_z = ev[2, :] / ev[0, :]
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(x, y, z, c="m", label="data")
    ax.plot(x_label, x_y[0]*x_label, x_z[0]*x_label, c="b", label=f"Contribution rate: {round_num(contribution[0], 3)}")
    ax.plot(x_label, x_y[1]*x_label, x_z[1]*x_label, c="g", label=f"Contribution rate: {round_num(contribution[1], 3)}")
    ax.plot(x_label, x_y[2]*x_label, x_z[2]*x_label, c="r", label=f"Contribution rate: {round_num(contribution[2], 3)}")
    ax.set_xlabel("X1", fontsize=20)
    ax.set_ylabel("X2", fontsize=20)
    ax.set_zlabel("X3", fontsize=20)
    plt.title(graph_title, fontsize=25)
    plt.legend()
    path = os.path.dirname(__file__)
    save_fname = os.path.join(path, "result", graph_title+"_3d.png")
    plt.savefig(save_fname)
    plt.tight_layout()
    plt.show()
    plt.close()
    press_data = data @ ev
    plt.scatter(press_data[:, 0], press_data[:, 1], c="m")
    plt.title(f"{graph_title}_press", fontsize=25)
    path = os.path.dirname(__file__)
    save_fname = os.path.join(path, "result", graph_title+"_press.png")
    plt.savefig(save_fname)
    plt.tight_layout()
    plt.show()
    plt.close()


def round_num(x, degit):
    x_degits = math.floor(math.log10(abs(x)))
    x_rounded = decimal.Decimal(str(x)).quantize(decimal.Decimal(str(10 ** (x_degits - degit + 1))), rounding = "ROUND_HALF_UP")
    return x_rounded


def plot_sum_con(data_dim, sum_con, press_dim, graph_title):
    plt.figure(figsize=(10,8))
    x = np.arange(1, data_dim+1, 1)
    plt.plot(x, sum_con[0:data_dim], c="b")
    plt.plot(press_dim, sum_con[press_dim], marker="o", c="r", markersize=12)
    plt.text(press_dim, sum_con[press_dim]-0.08, f"({press_dim}, {round_num(sum_con[press_dim], 2)})", fontsize=25)
    plt.title(graph_title+"_sum_con", fontsize=25)
    plt.xlabel("dimention", fontsize=20)
    plt.ylabel("sum contribution", fontsize=20)
    path = os.path.dirname(__file__)
    save_fname = os.path.join(path, "result", graph_title+"_sum_con.png")
    plt.savefig(save_fname)
    plt.tight_layout()
    plt.show()
    plt.close()


def main(args):
    fname = os.path.join("data", args.fname)
    data = open_csv(fname)
    s_data = standardize_data(data)
    eigenvector, contribution = pca(s_data)
    graph_title = os.path.splitext(args.fname)[0] 
    if data.shape[1] == 2:
        plot_2d(data, eigenvector, contribution, graph_title)
    elif data.shape[1] == 3:
        plot_3d(data, eigenvector, contribution, graph_title)
    else:
        goal = 0.9
        sum_con = np.zeros(data.shape[1])
        sum_con[0] += contribution[0]
        dim = 1
        for dim in range(0, data.shape[1]):
            sum_con[dim] = sum_con[dim-1] + contribution[dim]
            if (sum_con[dim-1]<goal) and (sum_con[dim]>goal):
                press_dim = dim + 1
        plot_sum_con(data.shape[1], sum_con, press_dim, graph_title)
        print("data_dim:", data.shape[1])
        print("press_dim:", press_dim)
        # ３次元以下に圧縮できたなら、プロット
        if dim == 2:
            plot_2d(data, eigenvector, sum_con)
        elif dim == 3:
            plot_3d(data, eigenvector, sum_con)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fname", help="csv file name")
    args = parser.parse_args()

    main(args)