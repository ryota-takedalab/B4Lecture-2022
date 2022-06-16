import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, n_clusters, filename, max_iter=1000) -> None:
        self.n_clusters = n_clusters
        self.filename = filename
        self.max_iter = max_iter

    def fit(self, x):
        # ランダムに最初のクラスタ点を決定
        tmp = np.random.choice(np.array(range(x.shape[0])))
        first_cluster = x[tmp]
        first_cluster = first_cluster[np.newaxis, :]

        # 最初のクラスタ点とそれ以外のデータ点との距離の２乗を計算し、それぞれをその総和で割る
        p = ((x - first_cluster) ** 2).sum(axis=1) / ((x - first_cluster) ** 2).sum()

        r = np.random.choice(np.array(range(x.shape[0])), size=1, replace=False, p=p)

        first_cluster = np.r_[first_cluster, x[r]]

        # cluster more than 3
        if self.n_clusters >= 3:
            while first_cluster.shape[0] < self.n_clusters:
                # 各クラスター点と各データポイントとの距離の２乗
                dist_f = (
                    (x[:, :, np.newaxis] - first_cluster.T[np.newaxis, :, :]) ** 2
                ).sum(axis=1)
                # 最も距離の近いクラスター点を算出
                f_argmin = dist_f.argmin(axis=1)
                # 最も距離の近いクラスターてんと各データポイントとの距離の２乗
                for i in range(dist_f.shape[1]):
                    dist_f.T[i][f_argmin != i] = 0

                # 新しいクラスタ点を確立的に導出
                pp = dist_f.sum(axis=1) / dist_f.sum()
                rr = np.random.choice(
                    np.array(range(x.shape[0])), size=1, replace=False, p=pp
                )
                # 新しいクラスター点を初期値として加える
                first_cluster = np.r_[first_cluster, x[rr]]

        # first label
        dist = ((x[:, :, np.newaxis] - first_cluster.T[np.newaxis, :, :]) ** 2).sum(
            axis=1
        )
        self.labels_ = dist.argmin(axis=1)
        labels_prev = np.zeros(x.shape[0])
        count = 0
        self.cluster_centers_ = np.zeros((self.n_clusters, x.shape[1]))

        images = [self.render_frame(x, first_cluster)]

        # clusters dont change or repeat max times
        while not (self.labels_ == labels_prev).all() and count < self.max_iter:
            # calculate clusters center
            for i in range(self.n_clusters):
                xx = x[self.labels_ == i, :]
                self.cluster_centers_[i, :] = xx.mean(axis=0)
            # calculate all each distance between DataPoints and ClusterCenters
            dist = (
                (x[:, :, np.newaxis] - self.cluster_centers_.T[np.newaxis, :, :]) ** 2
            ).sum(axis=1)
            # store previous cluster label
            labels_prev = self.labels_
            # after recalculating distance, allocate labels
            self.labels_ = dist.argmin(axis=1)
            count += 1
            self.count = count
            # add image into gif
            images.append(self.render_frame(x, self.cluster_centers_))

        images[0].save(
            f"{self.filename}.gif",
            save_all=True,
            append_images=images,
            optimize=False,
            duration=300,
            loop=0,
        )

    def render_frame(self, data, centroid):
        if data.shape[1] == 2:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            cmap = plt.get_cmap("tab10")
            for i in range(self.n_clusters):
                p = data[self.labels_ == i, :]
                ax.scatter(
                    p[:, 0], p[:, 1], marker="o", facecolor="None", edgecolors=cmap(i)
                )
            ax.scatter(
                centroid[:, 0],
                centroid[:, 1],
                s=100,
                linewidths=5,
                marker="x",
                c="k",
                label="centroid",
            )
            ax.set_title(self.filename, fontsize=20)
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$")
            ax.legend()

            buf = BytesIO()
            fig.savefig(buf, bbox_inches="tight", pad_inches=0.0)
            return Image.open(buf)
        elif data.shape[1] == 3:
            fig = plt.figure()
            plt.subplots_adjust(wspace=0.4)
            ax1 = fig.add_subplot(121, projection="3d")
            cmap = plt.get_cmap("tab10")
            for i in range(self.n_clusters):
                p = data[self.labels_ == i, :]
                ax1.scatter(
                    p[:, 0],
                    p[:, 1],
                    p[:, 2],
                    marker="o",
                    facecolor="None",
                    edgecolors=cmap(i),
                )
            ax1.scatter(
                centroid[:, 0],
                centroid[:, 1],
                centroid[:, 2],
                s=100,
                linewidths=5,
                marker="x",
                c="k",
                label="centroid",
            )
            ax1.set_title(self.filename, fontsize=20)
            ax1.set_xlabel("$x$")
            ax1.set_ylabel("$y$")
            ax1.set_zlabel("$z$")
            ax1.legend()

            ax2 = fig.add_subplot(122, projection="3d")
            for i in range(self.n_clusters):
                p = data[self.labels_ == i, :]
                ax2.scatter(
                    p[:, 0],
                    p[:, 1],
                    p[:, 2],
                    marker="o",
                    facecolor="None",
                    edgecolors=cmap(i),
                )
            ax2.scatter(
                centroid[:, 0],
                centroid[:, 1],
                centroid[:, 2],
                s=100,
                linewidths=5,
                marker="x",
                c="k",
                label="centroid",
            )
            ax2.view_init(elev=0, azim=0)
            ax2.set_title(self.filename, fontsize=20)
            ax2.set_xlabel("$x$")
            ax2.set_ylabel("$y$")
            ax2.set_zlabel("$z$")
            ax2.legend()

            buf = BytesIO()
            fig.savefig(buf, bbox_inches="tight", pad_inches=0.0)
            return Image.open(buf)
