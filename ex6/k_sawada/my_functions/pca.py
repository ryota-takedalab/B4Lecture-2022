import numpy as np


def pca(data):
    """primal component analysis

    Args:
        data (ndarray, axis=(each data, dimension)): input data

    Returns:
        ndarray, axis=(each pc): eigen values
        ndarray, axis=(each pc, dimension): eigen vectors
    """
    
    # interchangeable with my implementation
    # cov = np.cov(data.T, bias=True)
    avg = np.average(data, axis=0)
    cov = (data - avg).T @ (data - avg) / len(data)
    
    eigen_values, eigen_vectors = np.linalg.eig(cov)
    
    # sort by eigen_values
    order = eigen_values.argsort()[::-1]
    eigen_values = eigen_values[order]
    eigen_vectors = eigen_vectors.T[order]
    
    # adjust vector direction based on maximum absolute value to be positive
    to_reverse = \
        np.max(np.abs(eigen_vectors), axis=1) == np.max(eigen_vectors, axis=1)
    eigen_vectors = np.where(to_reverse, eigen_vectors * -1, eigen_vectors)
    
    # contribution rate
    contribution_rate = np.zeros(len(eigen_values))
    for i in range(len(eigen_values)):
        contribution_rate[i] = \
            np.sum(eigen_values[:i + 1]) / np.sum(eigen_values)
    print(contribution_rate)
    return eigen_values, eigen_vectors,


def dimension_compress(data, primal_component, variances,
                       contribution_rate=1.0, dimension=None):
    """dimension compression

    Args:
        data (ndarray, axis=(each data, dimension)): input data
        primal_component (ndarray): primal component by pca()
        variances (ndarray): variances by pca()
        contribution_rate (float, optional): contribution rate limit. Defaults to 1.0.
        dimension (int, optional): dimension limit. Defaults to None.

    Returns:
        ndarray, axis=(each data, dimension): compressed data
        list of float: contribution rate histories
    """
    # if dimension is not passed, data dimension is used.
    if dimension is None:
        dimension = data.shape[1]
    
    converted = np.empty((0, len(data)))
    total_contortion_rate = 0
    total_contribution_rate_histories = []
    
    # convert with primal components
    for d in range(dimension):
        if total_contortion_rate < contribution_rate:
            converted = np.append(converted, [data @ primal_component[d]], axis=0)
        total_contortion_rate += variances[d] / np.sum(variances)
        total_contribution_rate_histories.append(total_contortion_rate)
    return converted.T, total_contribution_rate_histories
