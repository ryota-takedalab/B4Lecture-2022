import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import argparse
from mpl_toolkits.mplot3d import Axes3D

def least_squares(x1,x2,dim):
    # Tの作成
    T = np.array([(x1**j*x2).sum() for j in range(dim+1)])

    # Aの作成
    A = pd.DataFrame()
    for j in range(dim+1):
        a = [(x1**(n+j)).sum() for n in range(dim+1)]
        A = pd.concat([A, pd.DataFrame([a])])

    # Aの逆行列を計算
    Ainv = np.linalg.inv(A)

    # 係数の解 w = Ainv * T
    w = np.dot(Ainv, T)

    # y座標の計算
    def predict_y(x):
        y=0
        for i, w_i in enumerate(w):
            y+=w_i * (x**i)
        return y

    return(predict_y, w)

def least_squares_3(x1, x2, x3, dim):
    # phiの作成
    n = len(x1)  # 100
    phi = np.zeros([n, 1 + dim*2])
    phi[:, :dim+1] = np.array([x1**n for n in range(dim+1)]).T
    phi[:, dim+1:] = np.array([x2**n for n in range(1,dim+1)]).T

    # Iの作成
    I = np.eye(1 + dim*2)
    w = np.dot(np.dot(np.linalg.pinv(np.dot(phi.T,phi)), phi.T), x3)
   
    # z座標の計算
    def predict_z(x, y):
        z = 0
        for i in range(dim+1): #<-x^0,x^1,x^2,y^1,y^2の係数
            z += w[i]*(x**i)
        for j in range(1, dim+1):
            z += w[j+dim]*(y**j)
        return z

    return predict_z, w

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('filepath', type=str, help='csv file name')
    parser.add_argument('--dim', type=int, default=1, help='x1 & x2 dimention number')
    args = parser.parse_args()
    filepath = args.filepath
    dim = args.dim
    name = filepath.split('.')[0]

    with open(filepath, encoding='utf8', newline='') as f:
        df = pd.read_csv(filepath).values
        x1 = df[:, :df.shape[1]-1]
        x2 = df[:, df.shape[1]-1:]

        if df.shape[1] == 2:
            print('df.shape[1] = 2')

            #多項式曲線フィッティング
            predict_y, w = least_squares(x1, x2, dim)

            df_w = pd.DataFrame()
            df_w = pd.concat([df_w, pd.DataFrame([w])])

            x = np.linspace(min(x1), max(x1), len(x1))
            y = predict_y(x)

            fig, ax = plt.subplots()           
            line_original = ax.plot(x1, x2, '.', label='Observed')
            line_approximation = ax.plot(x,y,color = 'orange')  # label = r'$f(x)=w[3]*x^3+w[2]*x^2+w[1]*x+w[0]$'
            ax.set(xlabel="$x_1$", ylabel="$x_2$", title=f'{name}' + f'  dim={dim}')
            ### 下の式をまとめて書けないか...n次式の時として
            # dim = 1  :1次式の時
            if dim == 1:
                ax.legend([line_original[0],line_approximation[0]],['original data',f'{w[1]:.2f}x+{w[0]:.2f}'],loc='upper left')
            # dim = 2  :2次式の時
            elif dim == 2:
                ax.legend([line_original[0],line_approximation[0]],['original data',f'y={w[2]:.2f}x^2+{w[1]:.2f}x+{w[0]:.2f}'],loc='upper left')
            # dim = 3  :3次式の時
            elif dim == 3:
                ax.legend([line_original[0],line_approximation[0]],['original data',f'y={w[3]:.2f}x^3+{w[2]:.2f}x^2+{w[1]:.2f}x+{w[0]:.2f}'],loc='upper left')

            plt.tight_layout()
            plt.savefig(f'{name}_dim{dim}.png')
            plt.show()
            plt.close()


        elif df.shape[1] == 3:
            x3 = x2
            x1 = df[:, :1]
            x2 = df[:, 1:2]
            print('data.shape[1] = 3')

            predict_z, w = least_squares_3(x1, x2, x3, dim)

            x = np.linspace(min(x1), max(x1), len(x1))
            y = np.linspace(min(x2), max(x2), len(x2))
            _x, _y = np.meshgrid(x, y)
            _z = predict_z(x, y)

            #plot            
            fig = plt.figure()
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.plot_wireframe(_x, _y, _z, linewidth=0.3, color = 'orange', label='Predected data')
            ax1.scatter(x1, x2, x3, label='Observed data')
            # ax1.view_init(elev=10, azim=4)  # elev: z, azim:x,y
            ax1.set(xlabel="$x_1$", ylabel="$x_2$", zlabel="$x_3$", title=f'{name}' + f'  dim={dim}')
            ax1.legend()
            
            ax2 = fig.add_subplot(122, projection='3d')
            ax2.scatter(x1, x2, x3, label='Observed data')
            # ax2.view_init(elev=10, azim=4)
            ax2.set(xlabel="$x_1$", ylabel="$x_2$", zlabel="$x_3$", title=f'{name}')
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(f'{name}_dim{dim}.png')
            # plt.savefig(f'{name}_dim{dim}_turned.png')
            plt.show()
            plt.close()

if __name__ == "__main__":
    main()