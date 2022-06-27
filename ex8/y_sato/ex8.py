import pickle

import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

class HMM:
    def __init__(self, data):
        """HMM
        Args:
            output(ndarray): output series
            answer_models(ndarray): answer labels
            PI(ndarray): initial probability array
            A(ndarray): transiton probability matrix
            B(ndarray): output probability matrix
        """

        self.output = np.array(data["output"])
        self.answer_models = np.array(data["answer_models"])
        self.PI = np.array(data["models"]["PI"])
        self.A = np.array(data["models"]["A"])
        self.B = np.array(data["models"]["B"])


    def forward_algorithm(self):
        """predict HMM by Forward Algorithm
        Returns:
            models (ndarray): predicted models
        """

        NUM, LEN = self.output.shape
        models = np.zeros(NUM)

        #「出力系列」の数だけやる
        for i in range(NUM):
            alpha = self.PI[:, :, 0] * self.B[:, :, self.output[i, 0]] #init
            for j in range(1, LEN):
                alpha = (
                    np.sum(alpha.T * self.A.T, axis=1).T
                    * self.B[:, :, self.output[i, j]] 
                )
            #確率が最大になるモデル
            models[i] = np.argmax(np.sum(alpha, axis=1))
        #最適なモデル(PI,A,B)が入ってるindexのリストを返す
        return models


    def viterbi_algorithm(self):
        """predict HMM by Virterbi algorithm
        Returns:
            model(ndarray): predict model
        """

        NUM, LEN = self.output.shape # 出力系列数、観測回数
        models = np.zeros(NUM)

        for i in range(NUM):
            alpha = self.PI[:, :, 0] * self.B[:, :, self.output[i, 0]]
            for j in range(1, LEN):
                alpha = (
                    np.max(alpha.T * self.A.T, axis=1).T
                    * self.B[:, :, self.output[i, j]]
                    )
            models[i] = np.argmax(np.max(alpha, axis=1))
        return models


    def calm_cm(self, models):
        """calculate confusion matrix
        Args:
            models(ndarray): predicted models
        Returns:
            cm(pandas.core.frame.DataFrame): confusion matrix
        """

        labels = list(set(self.answer_models))
        cm = confusion_matrix(self.answer_models, models, labels=labels)
        cm = pd.DataFrame(cm)
        return cm


    def calc_accuracy(self, models):
        """calculate accuracy
        Args:
            models(ndarray): predected models
        Returns:
            acc(float): accuracy rate[%]
        """

        acc = np.sum(self.answer_models == models) / self.answer_models.shape[0]
        return acc


def make_map(hmm):
    f_models = hmm.forward_algorithm()
    f_cm = hmm.calm_cm(f_models)
    f_acc = hmm.calc_accuracy(f_models)

    v_models = hmm.viterbi_algorithm()
    v_cm = hmm.calm_cm(v_models)
    v_acc = hmm.calc_accuracy(v_models)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    sns.heatmap(f_cm, cmap="Blues", annot=True, cbar=False, square=True)
    ax1.set(
        xlabel="Predicted model", ylabel="Actual model", title=f"Forwad Algorithm({filepath})\n acc: {f_acc}%\n"
    )
    ax2 = fig.add_subplot(1, 2, 2)
    sns.heatmap(v_cm, cmap="Blues", annot=True, cbar=False, square=True)
    ax2.set(
        xlabel="Predicted model", ylabel="Actual model", title=f"Virtebi Algorithm({filepath})\n acc: {v_acc}%\n"
    )
    plt.savefig("figs/" + f"{filepath}.png")
    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    filepaths = ["data1", "data2", "data3", "data4"]

    for filepath in filepaths:
        data = pickle.load(open(f"../{filepath}.pickle", "rb"))
        hmm = HMM(data)
        make_map(hmm)