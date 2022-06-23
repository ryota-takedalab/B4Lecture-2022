import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from io import BytesIO
from PIL import Image


def pca(data):
    """principal component analysis

    Args:
        data (np.ndarray): data extracted from csv file.

    Returns:
        np.ndarray: eigenvalues
        np.ndarray: eigenvector
        np.ndarray: standardized data
    """
    # standardization
    sc = StandardScaler()
    data_std = sc.fit_transform(data)

    # create a variance-covariance matrix
    cov_mat = np.cov(data_std.T)
    # get eigenvalues and eigenvectors of variance-covariance matrix
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    # get an array of indexes when sorting eigenvalues in descending-order
    sort_idxs = np.argsort(np.abs(eigen_vals))[::-1]
    # Sort eigenvalues in descending-order
    eigen_vals = eigen_vals[sort_idxs]
    # sort eigenvectors to correspond to the sorted eigenvalues(axis=1)
    eigen_vecs = eigen_vecs[:, sort_idxs]

    return eigen_vals, eigen_vecs, data_std


def render_frame(fig, ax, angle):
    """convert a 3D scatter plot into a PIL Image
    Args:
        fig (matplotlib.figure.Figure): figure data
        ax (list): list of axes objects
        angle (int): angle of rotation
    Returns:
        list: PIL Image
    """
    ax.view_init(30, angle)
    plt.close()
    # convert into a PIL Image
    buf = BytesIO()
    fig.savefig(buf, pad_inches=0.0)
    return Image.open(buf)


def main():
    # read csv files as DataFrame
    df_data1 = pd.read_csv("data1.csv", header=None)
    df_data2 = pd.read_csv("data2.csv", header=None)
    df_data3 = pd.read_csv("data3.csv", header=None)

    plt.rcParams["figure.figsize"] = (10, 7)

    # set drawing area for data1
    fig1, ax1 = plt.subplots()
    # draw a scatter plot of data1
    ax1.scatter(df_data1[0], df_data1[1], label="data1", c='m')
    ax1.set(title="data1", xlabel="x1", ylabel="x2")

    # set the data2 drawing area (3D)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(projection='3d')
    # set the elevation and azimuth of the axes in degrees
    ax2.view_init(elev=20, azim=105)
    # draw a scatter plot of data2
    ax2.plot(df_data2[0], df_data2[1], df_data2[2], marker="o", c='m', alpha=0.9, linestyle='None', label="data2")
    ax2.set(title="data2", xlabel="x1", ylabel="x2", zlabel="x3")
    ax2.legend()

    # convert DataFrame form to array form
    data1 = df_data1.to_numpy()
    data2 = df_data2.to_numpy()
    data3 = df_data3.to_numpy()

    # Data1
    data1_eigen_vals, data1_eigen_vecs, _ = pca(data1)
    data1_contrib = data1_eigen_vals/np.sum(np.abs(data1_eigen_vals))
    # plot
    xlist = np.linspace(np.min(data1[:, 0]), np.max(data1[:, 0]), 100)
    ylist_1 = xlist * (data1_eigen_vecs[0][1] / data1_eigen_vecs[0][0])
    ylist_2 = xlist * (data1_eigen_vecs[1][1] / data1_eigen_vecs[1][0])
    ax1.plot(xlist, ylist_1, color="r", label="Contribution rate: %f" % data1_contrib[0])
    ax1.plot(xlist, ylist_2, color="g", label="Contribution rate: %f" % data1_contrib[1])
    ax1.legend()

    # Data2
    data2_eigen_vals, data2_eigen_vecs, data2_std = pca(data2)
    data2_contrib = data2_eigen_vals/np.sum(np.abs(data2_eigen_vals))
    # plot
    cmap = plt.get_cmap("tab10")
    for i in range(len(data2_eigen_vecs)):
        xlist = np.linspace(np.min(data1[:, 0]), np.max(data1[:, 0]), 100)
        ylist = xlist * (data2_eigen_vecs[1][i] / data2_eigen_vecs[i][0])
        zlist = xlist * (data2_eigen_vecs[2][i] / data2_eigen_vecs[i][0])
        ax2.plot(xlist, ylist, zlist, color=cmap(i), label="Contribution rate: %f" % data2_contrib[i])
    ax2.legend()

    # compress Data2 into 2D
    # create a projection matrix
    W = np.stack([data2_eigen_vecs[0], data2_eigen_vecs[1]], axis=1)
    # dimensional compression (3D -> 2D)
    data2_pca = data2_std @ W
    # set drawing area for data2(compressed)
    fig3, ax3 = plt.subplots()
    # draw a scatter plot of data2(compressed)
    ax3.scatter(data2_pca[:, 0], data2_pca[:, 1], label="data2 (compressed)", c='m')
    ax3.set(title="data2 (compressed)", xlabel="PC1", ylabel="PC2")
    ax3.legend()

    # Data3
    data3_eigen_vals, data3_eigen_vecs, data3_std = pca(data3)
    data3_contrib = data3_eigen_vals/np.sum(np.abs(data3_eigen_vals))
    xlist = np.arange(0, len(data3_eigen_vals)-1, 1)
    ylist = np.cumsum(data3_contrib[:len(data3_eigen_vals)-1])
    # set drawing area for data2(compressed)
    fig4, ax4 = plt.subplots()
    ax4.plot(xlist, ylist)
    ax4.set(title="data4", xlabel="principal components", ylabel="cumulative contribution ratio")
    ax4.grid()

    # display changes in cumulative contribution ratio
    for i in range(0, len(data3_eigen_vals)-1):
        if ylist[i] > 0.9:
            ax4.vlines(i, ymin=0, ymax=ylist[i], colors='red', linestyle='dashed', linewidth=2)
            ax4.hlines(ylist[i], xmin=0, xmax=i, colors='red', linestyle='dashed', linewidth=2)
            ax4.text(i, ylist[i], "(x,y)=({}, {:.3f})".format(i, ylist[i]), va='top', size='large')
            break

    # save figures
    images = [render_frame(fig2, ax2, angle) for angle in range(360)]
    images[0].save('data2.gif', save_all=True, append_images=images[1:], duration=100, loop=0)
    plt.tight_layout()
    fig1.savefig("data1.png")
    fig2.savefig("data2.png")
    fig3.savefig("data2(compressed).png")
    fig4.savefig("data3.png")


if __name__ == "__main__":
    main()
