import argparse
import sys
import os

import matplotlib.animation as animation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random 

 
def read_csv(f_name):
    """
    ファイル読み込み

    parameter
    ---------
    f_name : str
        file name

    return
    ------
    data : numpy.ndarray
        csv data
    """
    path = os.path.dirname(__file__)
    f_name = os.path.join(path, "data", f_name)
    df = pd.read_csv(f_name)
    data = df.values

    return data


def scatter_2d(data, cen, save_fname, clu_num, clu, graph_title):
    """
    2次元散布図

    paramerters
    -----------
    data : numpy.ndarray
        csv data
    cen : np.ndarray
        center of cluster
    save : str
        save file name
    clu_num : int
        the number of cluster
    clu : numpy.ndarray
        cluster
    graph_title : str
        graph title
    """
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    cmap = plt.get_cmap("tab10")
    cen_x = cen[:, 0]
    cen_y = cen[:, 1]
    for i in range(clu_num):
        cdata = data[clu==i]
        x = cdata[:,0]
        y = cdata[:,1]
        #計算値とデータをプロット
        ax.plot(x, y, marker="o", linestyle="None", color=cmap(i))
    ax.plot(cen_x, cen_y, marker="x", markersize=12, markeredgewidth=5, linestyle="None", color="black")

    plt.title(graph_title, fontsize=25)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.tight_layout()
    if save_fname:
        path = os.path.dirname(__file__)
        save_fname = os.path.join(path, "result", save_fname)
        plt.savefig(save_fname)
    plt.show()


def scatter_3d(data, cen, save_fname, clu_num, clu, graph_title):
    """
    3次元散布図

    paramerters
    -----------
    data : numpy.ndarray
        csv data
    cen : numpy.ndarray
        center of cluster
    save_fname : str
        save file name
    clu_num : int
        the number of cluster
    clu : numpy.ndarray
        cluster
    graph_title : str
        graph title
    """
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("tab10")
    cen_x = cen[:, 0]
    cen_y = cen[:, 1]
    cen_z = cen[:, 2]
    #クラスタごとで描画
    for i in range(clu_num):
        cdata = data[clu==i]
        x = cdata[:,0]
        y = cdata[:,1]
        z = cdata[:,2]
        ax.plot(x, y, z, marker="o", linestyle='None', color=cmap(i))
    ax.plot(cen_x, cen_y, cen_z, marker="x", markersize=12, markeredgewidth=5, linestyle="None", color="black")
    plt.title(graph_title, fontsize=25)
    ax.set(
        xlabel="X",
        ylabel="Y",
        zlabel="Z"
    )
    def update(i):
        ax.view_init(elev=30.0, azim=3.6*i)
        return fig
    
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=100,
        interval=100
        )
    if save_fname:
        path = os.path.dirname(__file__)
        save_fname = os.path.join(path, "result", save_fname)
        ani.save(save_fname, writer="pillow")
    plt.tight_layout()
    plt.show()


def minimax(data, clu_num):
    """
    ミニマックス法

    paramerters
    -----------
    data : numpy.ndarray
        csv data
    clu_num : int
        the number of cluster
      
    return
    ------
    cen : numpy.ndarray
        center of cluster
    """
    num, dim = data.shape
    cidx = random.randint(0, num-1)
    dis = np.zeros((clu_num, num))
    cen = np.zeros((clu_num, dim))
    for k in range(clu_num):
        cen[k] = data[cidx]
        # 距離計算
        dis[k] =  np.sum((data-data[cidx])**2, axis=1)
        cidx = np.argmax(np.min(dis[:k+1], axis=0))

    return cen


def kplus(data, clu_num):
    """
    kmeans++

    paramerters
    -----------
    data : numpy.ndarray
        csv data
    clu_num : int
        the number of cluster
      
    return
    ------
    cen : numpy.ndarray
        center of cluster
    """
    num, dim = data.shape
    cidx = random.randint(0, num-1)
    dis = np.zeros((clu_num, num))
    cen = np.zeros((clu_num, dim))
    pr = np.zeros(num)
    for k in range(clu_num):
        cen[k] = data[cidx]
        # 距離計算
        dis[k] =  np.sum((data-data[cidx])**2, axis=1)
        # 確率作成
        pr = np.min(dis[:k+1], axis=0)
        pr = pr / np.sum(pr)
        # 次の中心
        cidx = np.random.choice(np.arange(num), 1, p=pr)

    return cen


def LGB(data, clu_num, cen):
    """
    LGB法

    paramerters
    -----------
    data : numpy.ndarray
        csv data
    clu_num : int
        the number of cluster
    cen : numpy.ndarray
        center of cluster
    
    return
    ------
    newcen : numpy.ndarray
        new center of cluster
    """
    # クラスタの中心をふたつに分ける
    delta = 0.01
    cen_b = cen-delta
    cen_a = cen+delta
    newcen = np.concatenate((cen_b, cen_a))
    M = newcen.shape[0]
    # kmeansアルゴリズム
    newcen, clu = kmean(data, M, newcen)
    if newcen.shape[0] >= clu_num:
        newcen = newcen[random.sample(range(len(newcen)), clu_num)]
        return newcen
    else:
        return LGB(data, clu_num, newcen)
        

def method(data, clu_num, mname):
    """
    初期値決定方法

    paramerters
    -----------
    data : numpy.ndarray
        csv data
    clu_num : int
        the number of cluster
    mname : str
        method name
      
    return
    ------
    cen :   numpy.ndarray
        center of cluster
    """
    assert mname in ["minimax", "kplus", "LGB"], "error method name"

    if mname == "minimax":
        cen = minimax(data, clu_num)
    elif mname == "kplus":
        cen = kplus(data, clu_num)
    else:
        if data.shape[1] == 2:
            ce = np.zeros((1, 2))
        elif data.shape[1] == 3:
            ce = np.zeros((1, 3))
        ce[0] = np.mean(data, axis=0)
        cen = LGB(data, clu_num, ce)

    return cen


def kmean(data, clu_num, cen):
    """
    kmeanアルゴリズム

    paramerters
    -----------
    data : numpy.ndarray
        csv data
    clu_num : int
        the number of cluster
    cen : numpy.ndarray
        center of cluster
    clu : numpy.ndarray
        cluster
      
    return
    ------
    newcen : numpy.ndarray
        new center of cluster
    clu : numpy.ndarray
        cluster
    """
    num, dim = data.shape
    dis = np.zeros((clu_num, num))
    newcen = np.zeros((clu_num, dim))
    while (True):
        for k in range(0, clu_num):
            # 距離計算
            r = np.sum((data-cen[k])**2, axis=1)
            # 距離保存
            dis[k] = r
        clu = np.argmin(dis, axis=0)
        for i in range(0, clu_num):            
            newcen[i] = data[clu==i].mean(axis=0)
        if np.allclose(cen,newcen):
            break
        cen = newcen
    return newcen, clu

def main(args):
    clu_num = args.clu_num
    save_fname = args.save_fname
    methodname = args.method
    data = read_csv(args.fname)
    # クラスタ計算&描画
    cen = method(data, clu_num, methodname)
    cen, clu = kmean(data, clu_num, cen)
    graphtitle = os.path.splitext(args.fname)[0]

    if data.shape[1] == 2:
        scatter_2d(data, cen, save_fname, clu_num, clu, graphtitle)
    
    elif data.shape[1] == 3:
        scatter_3d(data, cen, save_fname, clu_num, clu, graphtitle)

    else:
        print("error:over dimension")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fname", type=str)
    parser.add_argument("-cn", "--clu_num", type=int, default=4)
    parser.add_argument("-sf", "--save_fname", type=str)
    parser.add_argument("-m", "--method", type=str, default="minimax")
    args = parser.parse_args()

    main(args)