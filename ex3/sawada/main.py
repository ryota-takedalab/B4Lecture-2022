import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def regression_2d(data, title, dim=1,
                  regularize=False, reg_coef=1.0):
    data_regression = np.zeros((data.shape[0], dim + 1))
    density = 50
    x_linspace = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), density)
    x_plot = np.zeros((density, dim))
    
    for i in range(dim):
        data_regression[:, i] = np.power(data[:, 0], i + 1)
        x_plot[:, i] = np.power(x_linspace, i + 1)
    data_regression[:, dim] = data[:, 1]

    coefficient = linear_regression(data_regression, regularize, reg_coef)
    # plot initialization
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.subplots_adjust(hspace=0.6)

    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("y")
    ax.scatter(data[:, 0], data[:, 1], color="blue",
               label="Observed data")

    # regression result
    y_reg = np.concatenate([np.ones((len(x_plot), 1)), x_plot], 1) \
        @ coefficient

    label = "y = " + "{:.3g}".format(coefficient[0])
    for i in range(len(coefficient) - 1):
        label += " + " + "{:.3g}".format(coefficient[i + 1]) \
            + f"$x^{i+1}$"
    ax.plot(x_plot[:, 0], y_reg, color="red", label=label)
    ax.legend()
    plt.savefig(f"{title}_{dim}th_polynomial.png")
    plt.show()
        
        
def regression_3d(data, title, dim=1,
                  regularize=False, reg_coef=1.0):
    density = 50
    data_regression = np.zeros(
        (data.shape[0], ((dim + 1) * (dim + 2)) // 2))
    x1_linspace = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), density)
    x2_linspace = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), density)
    x1, x2 = np.meshgrid(x1_linspace, x2_linspace)
    x_plot = np.zeros((density, density, ((dim + 1) * (dim + 2)) // 2 - 1))
    
    col = 0
    for i in range(dim):
        for j in range(i + 2):
            data_regression[:, col] \
                = np.power(data[:, 0], (i + 1 - j)) \
                * np.power(data[:, 1], (j))
            x_plot[:, :, col] \
                = np.power(x1, (i + 1 - j)) \
                * np.power(x2, (j))
            col += 1
    data_regression[:, -1] = data[:, -1]

    coefficient = linear_regression(data_regression, regularize, reg_coef)
    # plot initialization
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='3d'))
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    fig.subplots_adjust(hspace=0.6)

    ax.set_title(title)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('y')
    ax.scatter(data.T[0], data.T[1], data.T[2],
               color="blue", label="Ovserved data")
    # regression result
    y_reg = np.concatenate([np.ones((density, density, 1)), x_plot], 2) \
        @ coefficient
    label = "y = " + "{:.3g}".format(coefficient[0])
    col = 1
    for i in range(dim):
        for j in range(i + 2):
            label += " + " + "{:.3g}".format(coefficient[col]) \
                + f"$x_1^{i + 1 - j}x_2^{j}$"
            col += 1
    ax.plot_wireframe(x1, x2, y_reg, color="red", label=label)
    ax.legend()
    plt.savefig(f"{title}_{dim}th_polynomial.png")
    plt.show()


def linear_regression(data, regularize=False, reg_coef=1.0):
    """linear regression

    Args:
        data (ndarray, size=(data_column, variables)): observation data.
        regularize (bool, optional): enable L2 regularize. Defaults to False.
        reg_coef (float, optional): regularize coefficient. Defaults to 1.0.

    Returns:
        ndarray: regression coefficient (intercept, slope, slope,...)
    """
    data_length = data.shape[0]
    data_dim = data.shape[1]

    # set regularize coefficient
    reg = 0
    if regularize:
        reg = reg_coef

    # add constant term
    x = np.concatenate([np.ones((data_length, 1)), data[:, :-1]], 1)

    parameters = \
        np.linalg.inv(x.T @ x - reg * np.identity(data_dim)) \
        @ x.T \
        @ data[:, -1]

    return parameters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ex3')
    parser.add_argument("-i", "--input", help="input file id", type=int)
    parser.add_argument("-d", '--dimension', type=int)
    parser.add_argument("-r", "--regularize", action="store_true")
    args = parser.parse_args()

    # read data
    data = pd.read_csv(f"../data{args.input}.csv").values

    if (data.shape[1] == 2):
        regression_2d(data, f"data{args.input}", dim=args.dimension,
                      regularize=args.regularize)
    if (data.shape[1] == 3):
        regression_3d(data, f"data{args.input}", dim=args.dimension,
                      regularize=args.regularize)
