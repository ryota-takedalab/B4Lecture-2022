import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import mpl_toolkits


def reg2d(x, y, deg=1, reg_coeff=1):

    """""
    2D-Regression 

    Args:
        x, y (float): x and y 2D array data.
        deg (int): order of regression.
        reg_coeff (float): Normalized Regression Coefficient.

    Returns:
        res_x (np.ndarray): x 2D regression of the Data.
        res_y (np.ndarray): y 2D regression of the Data.
        func (string): regression function.
    """""
    res_x = np.linspace(min(x), max(x), np.ceil((max(x)-min(x))/0.1).astype(int))
    res_y = np.zeros(len(res_x))
    phi = np.zeros([len(x), deg+1])
    for n in range(deg+1):
        phi[:, n] += x**n

    # Regularization
    i = np.eye(deg+1)
    inv = np.linalg.pinv(i * reg_coeff + np.dot(phi.T, phi))
    omega = np.dot(np.dot(inv, phi.T), y)

    # Regression
    for i in range(len(res_x)):
        for j in range(len(omega)):
            res_y[i] += (res_x[i] ** j) * omega[j]

    # Function
    func = f"{omega[0]:.2f}"
    for i in range(1, deg+1):
        func += f"+{omega[i]:.2f}x^{i}"
    func = f"$\\lambda={reg_coeff},y={func}$"

    return res_x, res_y, func


def reg3d(x, y, z, deg=1, reg_coeff=1):

    """""
    3D-Regression 

    Args:
        x, y, z (float): x, y, and z 3D array data.
        deg (int): order of regression (both x and y).
        reg_coeff (float): Normalized Regression Coefficient.

    Returns:
        res_x (np.ndarray): x 3D regression of the Data.
        res_y (np.ndarray): y 3D regression of the Data.
        res_z (np.ndarray): z 3D regression of the Data.
        func (string): regression function.
    """""
    # Setting up mesh grid and
    res_x = np.linspace(min(x), max(x), np.ceil((max(x)-min(x))/0.1).astype(int))
    res_y = np.linspace(min(y), max(y), np.ceil((max(x)-min(x))/0.1).astype(int))
    res_x, res_y = np.meshgrid(res_x, res_y)
    res_z = np.zeros([len(res_x), len(res_y)])

    phi = np.zeros([len(x), 2*deg+1])
    for n in range(deg+1):
        phi[:, n] += x ** n
    for m in range(deg):
        phi[:, m + deg + 1] += y ** (m + 1)

    i = np.eye(2*deg + 1)
    inv = np.linalg.pinv(i * reg_coeff + np.dot(phi.T, phi))
    omega = np.dot(np.dot(inv, phi.T), z)

    # Regression
    for i in range(0, deg + 1):
        res_z += (res_x ** i) * omega[i]
    for j in range(0, deg + 1):
        res_z += (res_y ** j) * omega[j + deg]

    # Function
    func = f'{omega[0]+omega[deg]:.2f}'
    for i in range(1, deg + 1):
        func += f"+{omega[i]:.2f}x^{i}"
    for i in range(1, deg + 1):
        func += f"+{omega[i+deg]:.2f}y^{i}"
    func = f"$\\lambda={reg_coeff},y={func}$"
    return res_x, res_y, res_z, func


def main():
    # Argparse
    parser = argparse.ArgumentParser(description='Name of the Data File')
    parser.add_argument('-fn', metavar='-f', dest='filename', type=str, help='Enter the Audio File Name',
                        required=True)
    parser.add_argument('-d', metavar='-1', dest='degree', type=int, help='Enter Polynomial Degree of Regression',
                        required=True)
    parser.add_argument('-l', metavar='-2', dest='lambd', type=float, help='Enter Normalized Regression Function',
                        required=True)
    args = parser.parse_args()

    # Reading Data and sample rate from audio, then convert the sample rate to 16kHz and channel to mono
    data = pd.read_csv(args.filename)

    # 2D Regression Plot
    if len(data.columns) == 2:
        res_x, res_y, func = reg2d(data['x1'], data['x2'], args.degree, args.lambd)
        # Scatter Plot
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(res_x, res_y, color='red', label=func)
        ax.scatter(data['x1'], data['x2'], color='blue', label='Data')
        ax.legend(loc='best')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title('2D Regression')
        ax.grid()
        plt.savefig(f'2D_Regression_lambda={args.lambd}.png')
        plt.show()
        plt.close()

    # 3D Regression Plot
    elif len(data.columns) == 3:
        res_x, res_y, res_z, func = reg3d(data['x1'], data['x2'], data['x3'], args.degree, args.lambd)
        fig = plt.figure(figsize=(12, 12))
        ax = mpl_toolkits.mplot3d.Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)
        ax.plot(data["x1"], data["x2"], data["x3"], marker="o", color='red', label='Data', linestyle='none')
        ax.plot_wireframe(res_x, res_y, res_z, color='green', label=func, alpha=0.4)
        ax.legend(loc='best')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('x3')
        ax.set_title('3D Regression', y=1)
        plt.savefig(f'3D_Regression_lambda={args.lambd}.png')
        plt.show()
        plt.close()


if __name__ == '__main__':
    main()
