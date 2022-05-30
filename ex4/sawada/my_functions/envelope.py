import numpy as np
import scipy.signal
import scipy.linalg

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
    return 20 * np.log10(np.abs(spectrum))


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


def envelope_lpc(data, p, fs):
    # TODO: docstring
    data = data * np.hanning(len(data))
    auto_correlation = f0.auto_correlation(data)
    alphas, e = levinson_durbin(auto_correlation, p)
    w, h = scipy.signal.freqz(np.sqrt(e), alphas, whole=True, fs=fs)
    
    return 20 * np.log10(np.abs(h))


def levinson_durbin(x, dimension):
    # TODO: docstring
    answers = np.zeros(dimension + 1)
    
    # base stage
    answers[0] = 1
    answers[1] = -x[1] / x[0]
    residual_error = x[0] + answers[1] * x[1]
    
    # recursive stage
    for i in range(1, dimension):
        # triple slice does not work
        # lambda_ = \
        #     -np.sum(answers[0:i + 1] * x[i + 1: 0: -1]) / residual_error
        lambda_ = -np.sum(answers[0:i + 2] *
                          np.flip(x[:i + 2])) / residual_error
        
        answers[:i + 2] += lambda_ * np.flip(answers[:i + 2])
        residual_error *= 1 - np.power(lambda_, 2)
        
    return answers, residual_error
