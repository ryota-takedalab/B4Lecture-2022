import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import argparse
from mpl_toolkits.mplot3d import Axes3D

def least_squares_2d(x1, x2, dim,l):
    """least squares regression
    
    Parameters:
    ----------
    x1 (ndarray) : input x1 data
    x2 (ndarray) : input x2 data
    dim (int) : dimention
    l (float) : regularization coefficient 
    
    Returns:
    ----------
    predect_y (def) : define y and linename
    w (ndarray) : weight

    """
    # Tの作成
    T = np.array([(x1**j*x2).sum() for j in range(dim+1)])

    # Aの作成
    A = pd.DataFrame()
    for j in range(dim+1):
        a = [(x1**(n+j)).sum() for n in range(dim+1)]
        A = pd.concat([A, pd.DataFrame([a])])

    # Aの逆行列を計算
    I = np.eye(dim+1)
    # Ainv = np.linalg.inv(A)  # 正則化なし
    Ainv = np.linalg.inv(A + l+I)  # 正則化あり

    # 係数の解 w = Ainv * T
    w = np.dot(Ainv, T)  # [1.08898963 2.0207883 ]

    # y座標の計算
    def predict_y(x):
        y=0
        linename = ''
        for i, w_i in enumerate(w):
            y+=w_i * (x**i)
            linename += f'{w[i]:.2f}x^{i}+'
        linename = linename[:-1]
        return y, linename

    return predict_y, w

def least_squares_3d(x1, x2, x3, dim, l):
    """least squares regression
    
    Parameters:
    ----------
    x1 (ndarray) : input x1 data
    x2 (ndarray) : input x2 data
    x3 (ndarray) : input x3 data
    dim (int) : dimention
    l (float) : regularization coefficient 
    
    Returns:
    ----------
    predect_y (def) : define z and linename
    w (ndarray) : weight

    """
    # phiの作成
    n = len(x1)  # 100
    phi = np.zeros([n, 1 + dim*2])
    phi[:, :dim+1] = np.array([x1**n for n in range(dim+1)]).T
    phi[:, dim+1:] = np.array([x2**n for n in range(1,dim+1)]).T

    # Iの作成
    I = np.eye(1 + dim*2)
    # w = np.dot(np.dot(np.linalg.pinv(np.dot(phi.T,phi)), phi.T), x3)  # 正則化なし
    w = np.dot(np.dot(np.linalg.inv(np.dot(phi.T,phi)+l*I), phi.T), x3)  # 正則化あり # [[-48.57517547][  2.12646582][ 29.59299973]]
    w = [x for row in w for x in row]  # wの１次元化
   
    # z座標の計算
    def predict_z(x, y):
        z = 0
        linename = ''
        for i in range(dim+1):  # <-x^0,x^1,x^2,y^1,y^2の係数
            z += w[i]*(x**i)
            linename += f'{w[i]:.2f}x^{i}+'
        for j in range(1, dim+1):
            z += w[j+dim]*(y**j)
            linename += f'{w[j+dim]:.2f}y^{j}+'
        linename = linename[:-1]
        return z, linename

    return predict_z, w

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('filepath', type=str, help='csv file name')
    parser.add_argument('--dim', type=int, default=1, help='x1 & x2 dimention number')
    parser.add_argument('--l', type=float, default=0, help='lambda')
    args = parser.parse_args()
    filepath = args.filepath
    dim = args.dim
    filename = filepath.split('.')[0]
    l = args.l

    with open(filepath, encoding='utf8', newline=''):
        df = pd.read_csv(filepath).values
        x1 = df[:, :df.shape[1]-1]
        x2 = df[:, df.shape[1]-1:]

        if df.shape[1] == 2:
            print('df.shape[1] = 2')

            # 多項式曲線フィッティング
            predict_y, w = least_squares_2d(x1, x2, dim, l)

            x = np.linspace(min(x1), max(x1), len(x1))
            y, linename = predict_y(x)

            fig, ax = plt.subplots()
            ax.plot(x1, x2, '.', label = 'Observed')
            ax.plot(x, y, color = 'orange', label = linename) 
            ax.set(xlabel="$x_1$", ylabel="$x_2$", title=f'{filename}' + f'  dim={dim}')

            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{filename}_dim{dim}.png')
            plt.show()
            plt.close()

        elif df.shape[1] == 3:
            print('data.shape[1] = 3')
            x3 = x2
            x1 = df[:, :1]
            x2 = df[:, 1:2]

            predict_z, w = least_squares_3d(x1, x2, x3, dim, l)

            x = np.linspace(min(x1), max(x1), len(x1))
            y = np.linspace(min(x2), max(x2), len(x2))
            _x, _y = np.meshgrid(x, y)
            _z, linename = predict_z(x, y)

            # plot     
            fig = plt.figure(figsize=(10,6))
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.plot_wireframe(_x, _y, _z, linewidth=0.3, color = 'orange', label=linename)
            ax1.scatter(x1, x2, x3, label='Observed data')
            ax1.view_init(elev=10, azim=4)  # elev: z, azim:x,y
            ax1.set(xlabel="$x_1$", ylabel="$x_2$", zlabel="$x_3$", title=f'{filename}' + f'  dim={dim}')
            ax1.legend()
            
            ax2 = fig.add_subplot(122, projection='3d')
            ax2.scatter(x1, x2, x3, label='Observed data')
            ax2.view_init(elev=10, azim=4)
            ax2.set(xlabel="$x_1$", ylabel="$x_2$", zlabel="$x_3$", title=f'{filename}')
            ax2.legend()
            
            plt.subplots_adjust(left=0.05, right=0.9, top=0.9, wspace=0.2, hspace=0.35)
            # plt.savefig(f'{filename}_dim{dim}.png')
            plt.savefig(f'{filename}_dim{dim}_turned.png')
            plt.show()
            plt.close()

if __name__ == "__main__":
    main()