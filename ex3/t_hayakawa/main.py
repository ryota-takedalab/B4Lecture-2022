import numpy as np
import argparse
import matplotlib.pyplot as plt


def regression_2d(x, y, dimension, lamda):
    """regression eqation

    Args:
        x (ndarray): input data
        y (ndarray): input data
        dimension (int): dimension
        lamda (float): regularizer coefficient

    Returns:
        ndarray, str: weight, equation name
    """
    phi = np.array([x**n for n in range(dimension+1)]).T
    iden_mat = np.eye(dimension+1)
    w = np.dot(np.dot(np.linalg.inv(np.dot(phi.T, phi)+lamda*iden_mat), phi.T), y)

    _x = np.linspace(x.min(), x.max(), 100)
    _y = np.array([np.dot(w, np.array([_x**n for n in range(dimension+1)]))])

    #equation name
    name = ""
    for i in range(dimension+1):
        if i > 0:
            if(i > 1):
                name += "+"+f"{w[i]:.3f}"+f"x^{{{i}}}"
            else:
                name += "+"+f"{w[i]:.3f}"+"x"
        else:
            name += f'{w[i]:.3f}'
    name = "$\lambda="+f"{lamda}"+",y="+name+"$"

    return _x, _y[0, :], name


def regression_3d(x0, x1, y, dimension, lamda):
    """regression equation

    Args:
        x0 (ndarray): input data
        x1 (ndarray): input data
        y (ndarray): input data
        dimension (int): dimension of regression
        lamda (float): regularizer coefficient 

    Returns:
        ndarray, str: weight, equation name
    """
    phi = np.zeros([x0.size, (dimension+1)*2])
    phi[:, :dimension+1] = np.array([x0**n for n in range(dimension+1)]).T
    phi[:, dimension+1:] = np.array([x1**n for n in range(dimension+1)]).T
    iden_mat = np.eye((dimension+1)*2)
    w = np.dot(np.dot(np.linalg.pinv(
        np.dot(phi.T, phi)+lamda*iden_mat), phi.T), y)

    #equation name
    name = ""
    for i in range(dimension+1):
        if i > 0:
            if(i > 1):
                name += "+"+f"{w[i]:.3f}"+f"x_0^{{{i}}}"
            else:
                name += "+"+f"{w[i]:.3f}"+f"x_0"
    for i in range(dimension+1):
        if i > 0:
            if(i > 1):
                name += "+"+f"{w[i+dimension+1]:.3f}"+f"x_1^{{{i}}}"
            else:
                name += "+"+f"{w[i+dimension+1]:.3f}"+f"x_1"

    name = f'{w[0]+w[dimension+1]:.3f}'+name
    name = "$\lambda="+f"{lamda}"+",y="+name+"$"

    return w, name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="input filename .csv")
    parser.add_argument("--dim", required=True, type=int, help="dimension")
    parser.add_argument("--lamda", type=float, default=0)

    args = parser.parse_args()

    #load text from .csv
    data = np.loadtxt(args.filename+".csv", delimiter=",", skiprows=1)

    #split data
    x = data[:, :-1]
    y = data[:, -1]

    if x.shape[1] == 1:
        _x, _y, name = regression_2d(x[:, 0], y, args.dim, args.lamda)

        #plot
        fig = plt.figure(figsize=(8, 6))
        plt.plot(_x, _y, c="r", label=name)
        plt.scatter(x[:, 0], y, label="Observed data")
        plt.title(args.filename)
        plt.xlabel("$x_0$")
        plt.ylabel("$y$")
        plt.legend()
        plt.show()

    if x.shape[1] == 2:
        w, name = regression_3d(x[:, 0], x[:, 1], y, args.dim, args.lamda)

        #calcurate "y" using weight
        x0 = np.linspace(x[:, 0].min(), x[:, 0].max(), 100)
        x1 = np.linspace(x[:, 1].min(), x[:, 1].max(), 100)
        _x0, _x1 = np.meshgrid(x0, x1)
        _y = np.zeros(_x0.shape)
        for i in range(args.dim+1):
            _y += w[i]*(_x0**i)+w[i+args.dim+1]*(_x1**i)

        #plot
        fig = plt.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(121, projection="3d")
        ax1.plot_wireframe(_x0, _x1, _y, linewidth=0.2,
                           color="red", label=name)
        ax1.scatter(x[:, 0], x[:, 1], y, label="Observed data")
        ax1.set_title(args.filename)
        ax1.set_xlabel("$x_0$")
        ax1.set_ylabel("$x_1$")
        ax1.set_zlabel("$y$")
        ax1.legend(bbox_to_anchor=(1, 1.03))

        ax2 = fig.add_subplot(122, projection="3d")
        ax2.plot_wireframe(_x0, _x1, _y, linewidth=0.2, color="red")
        ax2.scatter(x[:, 0], x[:, 1], y)
        ax2.view_init(elev=0, azim=0)
        ax2.set_title(args.filename)
        ax2.set_xlabel("$x_0$")
        ax2.set_ylabel("$x_1$")
        ax2.set_zlabel("$y$")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
