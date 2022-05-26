import argparse
import os
import math
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sympy import latex, Symbol
import decimal

def regression_2d(x, y, deg, lam):
    phi = np.array([[p ** i for i in range(deg + 1)] for p in x])
    print("phi:", phi)
    a = np.linalg.inv(phi.T @ phi + lam * np.eye(deg + 1)) @ phi.T @ y
    print("a:", a)
    return a

def regression_3d(x, y, z, deg_x, deg_y, lam):
    phi_x = np.array([[p ** i for i in range(deg_x + 1)] for p in x])
    phi_y = np.array([[p ** (i + 1) for i in range(deg_y)] for p in y])
    phi = np.hstack([phi_x, phi_y])
    a = np.linalg.inv(phi.T @ phi + lam * np.eye(deg_x + deg_y + 1)) @ phi.T @ z
    return a

def round_num(x, degit):
    # print("x:", x)
    x_degits = math.floor(math.log10(abs(x)))
    # print("x_degits:", x_degits)
    x_rounded = decimal.Decimal(str(x)).quantize(decimal.Decimal(str(10 ** (x_degits - degit + 1))), rounding = "ROUND_HALF_UP")
    return x_rounded

# 数式を表示するための関数
def latexfunc(a, deg_x, deg_y = None):
    x = Symbol("x")
    f = 0
    for i in range(deg_x + 1):
        f += round_num(a[i], 2) * x ** i
    if deg_y is not None:
        y = Symbol("y")
        for i in range(deg_y):
            f += round_num(a[deg_x + i + 1], 2) * y ** (i + 1)
    f = latex(f)
    return f

def my_removesuffix(str, suffix):
    return str[: -len(suffix)] if str.endswith(suffix) else str

def main(args):
    fname = args.fname
    save_fname = args.save_fname
    deg_x = args.deg_x
    deg_y = args.deg_y
    lam = args.lam

    # import data
    path = os.path.dirname(os.path.abspath(__file__))
    graphtitle = my_removesuffix(fname, ".csv")
    fname = os.path.join(path, "data", fname)
    data = pd.read_csv(fname).values
    if save_fname != None:
        save_fname = os.path.join(path, "result", save_fname)

    if data.shape[1] == 2:
        x = data[:, 0]
        y = data[:, 1]

        reg_x = np.linspace(x.min(), x.max(), 500)
        reg_y = np.zeros_like(reg_x)
        a = regression_2d(x, y, deg_x, lam)

        y_hat = np.zeros_like(x)
        for i in range(len(a)):        
            reg_y += a[i] * reg_x ** i
            y_hat += a[i] * x ** i
        # 平均二乗誤差
        mse = round_num(np.mean((y - y_hat) ** 2), 2)
        print("mse:", mse)
        fig = plt.figure()
        ax = fig.add_subplot(111, xlabel = "X", ylabel = "Y")
        ax.scatter(x, y, s = 12, c = "darkblue", label = "observed")
        plt.plot(reg_x, reg_y, c = "r", label = "predicted")
        ax.grid(ls = "--")
        ax.set_title(
            graphtitle
            + "  (deg = {0}, lam = {1}) MSE = {2:.3f}\n".format(deg_x, lam,mse )
            + "$f(x) = "
            + latexfunc(a, deg_x)
            +"$"
        )
        ax.legend(loc = "best", fontsize = 10)
        if save_fname != None:
            plt.savefig(save_fname)
        plt.show()

    elif data.shape[1] == 3:
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]

        reg_x = np.linspace(x.min(), x.max(), 30)
        reg_y = np.linspace(y.min(), y.max(), 30)
        reg_x, reg_y = np.meshgrid(reg_x, reg_y)
        reg_z = np.zeros_like(reg_x)
        a = regression_3d(x, y, z, deg_x, deg_y, lam)
        z_hat = np.zeros_like(x)
        for i in range(deg_x + 1):
            reg_z += a[i] * reg_x **i
            z_hat += a[i] * x ** i
        for i in range(deg_y):
            reg_z += a[deg_x + i + 1] * reg_y ** (i + 1)
            z_hat += a[deg_x + i] * y ** (i + 1)
        mse = round(np.mean((z - z_hat) ** 2), 3)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection = "3d")
        ax.scatter3D(x, y, z, s = 20, c = "darkblue", label = "observed")
        ax.plot_wireframe(
            reg_x, reg_y, reg_z, color = "red", linewidth = 0.5, label = "predicted"
        )
        ax.set(
            title = graphtitle
            + "_3D  (deg_x = {0}, deg_y = {1}, lam = {2})  MSE = {3:.3f}\n".format(
                deg_x, deg_y, lam, mse
            )
            + "$f(x, y) = "
            + latexfunc(a, deg_x, deg_y)
            + "$",
            xlabel = "X",
            ylabel = "Y",
            zlabel = "Z",
        )
        ax.legend(loc = "best", fontsize = 10)
        if save_fname != None:
            plt.savefig(save_fname.replace("gif", "png"))

        def update(i):
            ax.view_init(elev = 30.0, azim = 3.6 * i)
            return fig
        
        ani = animation.FuncAnimation(fig, update, frames = 100, interval = 100)
        if save_fname != None:
            ani.save(save_fname, writer = "pillow")
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Regression and Regularization.")
    parser.add_argument("fname", type = str, help = "Load Filename")
    parser.add_argument(
        "-s",
        "--save_fname",
        type = str,
        help = "Wave Filename",
        )
    parser.add_argument(
        "-x",
        "--deg_x",
        type = int,
        help = "Degree for x in regression function",
        required = True,
    )
    parser.add_argument(
        "-y",
        "--deg_y",
        type = int,
        help = "Degree for y in regression function (optional, Default = 0).\nif you load data3.csv, this is required.",
        default = 0,
    )
    parser.add_argument(
        "-l",
        "--lam",
        type = float,
        help = "Normalization coefficient ( optional, Default = 0).",
        default = 0,
    )
    args = parser.parse_args()
    main(args)