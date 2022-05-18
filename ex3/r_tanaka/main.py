import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# data1とdata2の描画領域を設定
plt.rcParams["figure.figsize"] = (12, 10)
fig, ax = plt.subplots(2, 1)
fig.subplots_adjust(hspace=0.5)

# read csv files
data1 = pd.read_csv("data1.csv")
data2 = pd.read_csv("data2.csv")
data3 = pd.read_csv("data3.csv")

# data1とdata2の散布図を描画
ax[0].scatter(data1['x1'], data1['x2'])
ax[0].set(title="data1", xlabel="x1", ylabel="x2")

ax[1].scatter(data2['x1'], data2['x2'])
ax[1].set(title="data2", xlabel="x1", ylabel="x2")

# data3の描画領域（３次元）を設定
fig3 = plt.figure()
ax3 = Axes3D(fig3)

# data3の散布図を描画
ax3.plot(data3['x1'], data3['x2'], data3['x3'], marker="o", linestyle='None')
ax3.set(title="data3", xlabel="x1", ylabel="x2", zlabel="x3")


plt.show()
fig.savefig("data1&2.png")
fig3.savefig("data3.png")
