import numpy as np


class PCA:
    def __init__(self, data, is_std=True) -> None:
        if is_std:
            self.data = self.standardize(data)
        else:
            self.data = data

    def standardize(self, data):
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    def pca(self):
        # calculate covariance matrix
        # 共分散行列
        cov_mat = np.cov(self.data.T)

        # eigenvalue and eigenvector
        # 固有値、固有ベクトル
        eig_val, eig_vec = np.linalg.eig(cov_mat)

        # sort eigenvalue and eigenvector in descending order
        sort_eig_val = np.sort(eig_val)[::-1]
        sort_eig_vec = eig_vec[:, np.argsort(eig_val)[::-1]]

        # contribution rate
        cont_rate = sort_eig_val / np.sum(sort_eig_val)

        return sort_eig_val, sort_eig_vec, cont_rate
