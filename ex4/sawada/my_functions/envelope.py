import numpy as np
import scipy.signal
import scipy.linalg
import matplotlib.pyplot as plt

from .import cepstrum
from . import f0


def log_spectrum(data):
    """log spectrum

    Args:
        data (ndarray, axis=(time, )): input data

    Returns:
        ndarray, axis=(frequency, ): log amplitude
    """
    spectrum = np.fft.fft(data * np.hanning(len(data)))
    return np.log10(np.abs(spectrum))


def envelope_cepstrum(data):
    """spectrum envelope based on cepstrum

    Args:
        data (ndarray, axis=(time, )): input data

    Returns:
        ndarray, axis=(frequency, ): spectram envelope based on cepstrum
    """
    data_cepstrum = cepstrum.cepstrum(data)
    lp_lifter = cepstrum.craete_lifter(len(data), cutoff_frame=20)
    return np.real(np.fft.fft(data_cepstrum * lp_lifter))


# TODO: おかしい
def envelope_lpc(data, p, fs):
    # TODO: docstring
    length = len(data)
    auto_correlation = f0.auto_correlation(data)
    
    toeplitz = np.zeros((length, length))
    for i in range(length):
        toeplitz[i:, i] = auto_correlation[i:]
        toeplitz[i, i:] = auto_correlation[i:]
    
    # alphas = solve_toeplitz(toeplitz, auto_correlation)
    alphas = scipy.linalg.solve_toeplitz(auto_correlation, -auto_correlation)
    
    w, h = scipy.signal.freqz(1, alphas, fs=fs)
    
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111)
    # ax.plot(log_spectrum(h))
    # fig.show()
    return log_spectrum(h)


# TODO: おかしい
def solve_toeplitz(x, b):
    if (x.shape[0] != x.shape[1] or x.shape[1] != len(b)):
        raise ValueError("wrong shape")
    dimension = len(b)
    answers = np.zeros(dimension)
    residual_error = np.zeros(dimension)
    
    # base stage
    answers[0] = -1 * x[1, 1] / x[0, 0]
    residual_error[0] = x[0, 0] + answers[0] * x[1, 1]
    
    # recursive stage
    for i in range(dimension - 1):
        lambda_ = 0
        for j in range(i + 2):
            lambda_ -= answers[j] * b[i + 1 - j]
        lambda_ /= residual_error[i]
        
        answers[: i + 2] = (np.concatenate([[1], answers[: i + 1], [0]]) +
                            lambda_ *
                            np.concatenate([[0], answers[: i + 1], [1]]))[1:]
        residual_error[i + 1] = (1 - np.power(lambda_, 2)) * residual_error[i]
    
    return answers
