import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from io import BytesIO
from PIL import Image

from pca import PCA


def render_frame(df, x_label, coef_y, coef_z, filename, cont_rate, angle):
    fig = plt.figure()
    ax = fig.add_subplot(
        111, projection="3d", title=filename, xlabel="x1", ylabel="x2", zlabel="x3",
    )
    ax.scatter(df[:, 0], df[:, 1], df[:, 2], label="data")

    cmap = plt.get_cmap("tab10")

    for i in range(df.shape[1]):
        plt.plot(
            x_label,
            coef_y[i] * x_label,
            coef_z[i] * x_label,
            label=f"Contribution rate: {round(cont_rate[i], 3)}",
            c=cmap(i + 1),
        )
    ax.view_init(30, angle)
    plt.close()
    ax.legend(bbox_to_anchor=(1, 1), loc="upper right")

    buf = BytesIO()
    fig.savefig(buf)
    return Image.open(buf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="data file .csv")

    args = parser.parse_args()

    # load text from .csv
    df = pd.read_csv(args.filename + ".csv", header=None).values

    pca_ = PCA(df, True)
    _, eig_vec, cont_rate = pca_.pca()

    cmap = plt.get_cmap("tab10")

    if df.shape[1] == 2:
        x_label = np.linspace(np.min(df[:, 0]), np.max(df[:, 0]), 100)
        coef = eig_vec[1] / eig_vec[0]

        fig = plt.figure()
        ax = fig.add_subplot(111, title=args.filename, xlabel="x1", ylabel="x2",)
        ax.scatter(df[:, 0], df[:, 1], label="data")

        for i in range(df.shape[1]):
            plt.plot(
                x_label,
                coef[i] * x_label,
                label=f"Contribution rate: {round(cont_rate[i],3)}",
                c=cmap(i + 1),
            )
        plt.grid(ls="--")
        plt.legend()
        plt.savefig(f"{args.filename}_pca.png")
        plt.show()

    elif df.shape[1] == 3:
        x_label = np.linspace(np.min(df[:, 0]), np.max(df[:, 0]), 100)
        coef_y = eig_vec[1] / eig_vec[0]
        coef_z = eig_vec[2] / eig_vec[0]

        images = [
            render_frame(df, x_label, coef_y, coef_z, args.filename, cont_rate, angle)
            for angle in range(0, 360, 2)
        ]
        images[0].save(
            f"{args.filename}_pca.gif",
            save_all=True,
            append_images=images[1:],
            duration=100,
            loop=0,
        )

        data_pca = pca_.data @ eig_vec
        fig = plt.figure()
        fig.add_subplot(111, title=f"{args.filename}_2d", xlabel="PC1", ylabel="PC2")
        plt.scatter(data_pca[:, 0], data_pca[:, 1], label="data")
        plt.grid(ls="--")
        plt.legend()
        plt.show()

    else:
        cum_cont_rate = np.cumsum(cont_rate)
        point = [np.min(np.where(cum_cont_rate >= 0.9)), 0]
        point[1] = cum_cont_rate[point[0]]
        point[0] += 1

        fig = plt.figure()
        fig.add_subplot(
            111,
            title=f"{args.filename}_cumulative contribution rate",
            xlabel="Dimension",
            ylabel="Cumulative contribution rate",
            xticks=np.append(np.linspace(0, 100, 6), point[0]),
            yticks=np.append(np.linspace(0, 1, 6), point[1]),
        )
        plt.plot(range(1, len(cont_rate) + 1), cum_cont_rate, c="darkblue")
        plt.axvline(x=point[0], c="r", linestyle="dashed")
        plt.axhline(y=point[1], c="r", linestyle="dashed")
        plt.grid(ls="--")
        plt.savefig(f"{args.filename}_cont.png")
        plt.show()


if __name__ == "__main__":
    main()
