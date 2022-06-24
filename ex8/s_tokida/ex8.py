import argparse
import pickle
import time

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from model import HMM


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("filepath", choices=["data1", "data2", "data3", "data4"])
    args = parser.parse_args()

    filepath = args.filepath

    # load pickle data
    data = pickle.load(open(f"../{filepath}.pickle", "rb"))
    output = np.array(data["output"], dtype="int8")
    answer_models = np.array(data["answer_models"], dtype="int8")
    PI = np.array(data["models"]["PI"])
    A = np.array(data["models"]["A"])
    B = np.array(data["models"]["B"])

    # data
    # ├─answer_models  # Answer labels  (100,)
    # ├─output  # Output series  (100, 100)
    # └─models  # HMM models
    #     ├─PI  # Initial probability  (5, 3, 1) / (5, 5, 1)
    #     ├─A  # State transition probability  (5, 3, 3) / (5, 5, 5)
    #     └─B  # Output probability  (5, 3, 3) / (5, 5, 5)

    hmm = HMM(output, answer_models, PI, A, B)

    # forward algorithm
    f_start = time.time()
    f_models = hmm.forward_algorithm()
    f_end = time.time()
    f_cm = hmm.calc_cm(f_models)
    f_acc = hmm.calc_accuracy(f_models)

    # viterbi_algorithm
    v_start = time.time()
    v_models = hmm.viterbi_algorithm()
    v_end = time.time()
    v_cm = hmm.calc_cm(v_models)
    v_acc = hmm.calc_accuracy(v_models)

    # plot
    fig = plt.figure(figsize=(8, 6))

    ax1 = fig.add_subplot(1, 2, 1)
    sns.heatmap(f_cm, cmap="binary", annot=True, cbar=False, square=True, ax=ax1)
    ax1.set(
        xlabel="Predicted model",
        ylabel="Actual model",
        title=f"Forward Algorithm({filepath})\n acc: {f_acc}%\n time: {f_end-f_start:.4f}",
    )
    ax2 = fig.add_subplot(1, 2, 2)
    sns.heatmap(v_cm, cmap="binary", annot=True, cbar=False, square=True, ax=ax2)
    ax2.set(
        xlabel="Predicted model",
        ylabel="Actual model",
        title=f"Viterbi Algorithm({filepath})\n acc: {v_acc}%\n time: {v_end-v_start:.4f}",
    )
    plt.savefig("figs/" + f"{filepath}_heatmap.png")
    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
