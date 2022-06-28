import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


class HMM:
    def __init__(self, output, answer_models, PI, A, B):
        """HMM
        Args:
            output (ndarray):output series
            answer_models (ndarray):answer labels
            PI (ndarray):initial probability array
            A (ndarray):transition probability matrix
            B (ndarray):output probability matrix
        """

        self.output = output
        self.answer_models = answer_models
        self.PI = PI
        self.A = A
        self.B = B

    def forward_algorithm(self):
        """predict HMM by Forward algorithm
        Returns:
            models (ndarray):predicted models
        """

        NUM, LEN = self.output.shape
        models = np.zeros(NUM)

        for i in range(NUM):
            alpha = self.PI[:, :, 0] * self.B[:, :, self.output[i, 0]]  # init
            for j in range(1, LEN):
                alpha = (
                    self.B[:, :, self.output[i, j]]
                    * np.sum(self.A.T * alpha.T, axis=1).T
                )
            models[i] = np.argmax(np.sum(alpha, axis=1))
        return models

    def viterbi_algorithm(self):
        """predict HMM by Viterbi algorithm
        Returns:
            models (ndarray):predicted models
        """

        NUM, LEN = self.output.shape  # 出力系列数, 観測回数 (100, 100)
        models = np.zeros(NUM)

        for i in range(NUM):
            alpha = self.PI[:, :, 0] * self.B[:, :, self.output[i, 0]]  # init
            for j in range(1, LEN):
                alpha = (
                    self.B[:, :, self.output[i, j]]
                    * np.max(self.A.T * alpha.T, axis=1).T
                )
            models[i] = np.argmax(np.max(alpha, axis=1))
        return models

    def calc_cm(self, models):
        """calculate confusion matrix
        Args:
            models (ndarray): predected models
        Returns:
            cm (pandas.core.frame.DataFrame): confusion matrix
        """
        # print(self.answer_models)  いっぱいあるから、set型にしてる
        labels = list(set(self.answer_models))
        labels = list(map(lambda x: x + 1, labels))

        cm = confusion_matrix(self.answer_models, models, labels=labels)
        cm = pd.DataFrame(cm, index=labels, columns=labels)

        return cm

    def calc_accuracy(self, models):
        """calculate accuracy
        Args:
            models (ndarray): predected models
        Returns:
            acc (float): accracy rate[%]
        """

        acc = np.sum(self.answer_models == models) / self.answer_models.shape[0] * 100
        return acc
