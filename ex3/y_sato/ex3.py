import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import axes3d


class LinearModel:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.a = []
        self.name = ""


    #任意の次数(1を含む)の多項式に変数変換
    def polynomial(self, x, degs):
        arr_x = np.array(x)

        #0,1乗の列をresultに追加
        result = np.insert(arr_x, 0, 1, axis=1)

        #d乗の列を追加
        for i in range(1,degs):
            x_degs = arr_x ** (i+1)
            result = np.hstack([result, x_degs])

        return result


    def fit(self, degs, lam=0.1):
        arr_x = self.polynomial(self.x, degs=degs)
        arr_y = self.y
        i = np.eye(arr_x.shape[1])
        a = np.linalg.inv(arr_x.T @ arr_x + lam * i) @ arr_x.T @ arr_y
        self.a = a

        return a


    def predict2d(self, x1):
        a = self.a
        Z = a[0][0] * x1 ** 0 #もっと良い方法あるはず
        name = "$x_{2}="
        name += f"{self.a[0][0]:.2f}"

        for i in range(1, len(a)):
            Z += a[i] * x1 ** i
            name += f"{a[i][0]:+.2f}x_{1}^{i}"
        name += "$"
        self.name = name

        return Z


    def predict3d(self, x1, x2):
        a = self.a
        Z = a[0][0] * x1 ** 0
        name = "$x_{3}="
        name += f"{self.a[0][0]:.2f}"

        for i in range(1, len(a), 2):
            Z += a[i][0] * x1 ** ((i+1)/2) + a[i+1][0] * x2 ** ((i+1)/2)
            name += f"{a[i][0]:+.2f}x_{1}^{int((i+1)/2)}{a[i+1][0]:+.2f}x_{2}^{int((i+1)/2)}"
        name += "$"
        self.name = name

        return Z


    def label(self):
        return self.name



def regression2d(data, title, deg):
    data_size = data.shape[1]
    x = data[:, 0 : data_size - 1]
    y = data[:, data_size - 1 :]

    #generate model
    model = LinearModel(x, y)

    #model fitting
    a = model.fit(deg)

    #plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, y, label="Observed Value")
    x1 = np.linspace(x.min(), x.max())
    predict = model.predict2d(x1)
    ax.plot(x1, predict, label=model.label(), color="red")
    ax.set(xlabel="$x_1$", ylabel="$x_2$", title=title)
    ax.grid()
    ax.legend()
    plt.show()
    plt.close()


def regression3d(data, title, deg):
    data_size = data.shape[1]
    x = data[:, 0 : data_size - 1]
    y = data[:, data_size - 1 :]

    #generate model
    model = LinearModel(x, y)

    #model fitting
    a = model.fit(deg)

    #plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter3D(x[:, 0], x[:, 1], y, c="r", label="Observed data")
    X, Y = np.meshgrid(np.arange(x[:, 0].min(), x[:, 0].max(), 0.1), np.arange(x[:, 1].min(), x[:, 1].max(), 0.1))
    Z = model.predict3d(X, Y)
    ax.plot_wireframe(X, Y, Z, label=model.label(), alpha=0.5)
    ax.set(xlabel="$x_1$", ylabel="$x_2$", zlabel="$x_3$", title=title)
    ax.legend()
    plt.show()
    plt.close()


if __name__ == "__main__":

    data1 = pd.read_csv('data1.csv', delimiter=',').values
    data2 = pd.read_csv('data2.csv', delimiter=',').values
    data3 = pd.read_csv('data3.csv', delimiter=',').values

    regression2d(data1, "data1", 1)
    regression2d(data2, "data2", 3)
    regression3d(data3, "data3", 2)
