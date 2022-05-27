import numpy as np

from . import cepstrum


def f0_estimate_cepstrum(data, fs, nperseg=512, noverlap=256):
    """f0 estimation based on cepstrum

    Args:
        data (ndarray, axis=(time, )): input data
        fs (int): sampling rate
        nperseg (int, optional): samples in each time segment. Defaults to 512.
        noverlap (int, optional): samples of segment overlap. Defaults to 256.

    Returns:
        ndarray, axis=(time, ): f0 array
    """
    # cepstrum
    cepstrums = cepstrum.st_cepstrum(data, fs, nperseg, noverlap)
    
    # liftering
    liftered_cepstrums = cepstrums \
        * cepstrum.craete_lifter(nperseg, mode="hp")
    
    # get peak freaquency
    peak_indexes = np.argmax(liftered_cepstrums[:, :nperseg // 2], axis=1)
    peak_frequencies = fs / peak_indexes
    
    return peak_frequencies


def f0_estimate_autocorrelation(data, fs, nperseg=512, noverlap=256):
    """f0 estimation based on autocorrelation

    Args:
        data (ndarray, axis=(time, )): input data
        fs (int): sampling rate
        nperseg (int, optional): samples in each time segment. Defaults to 512.
        noverlap (int, optional): samples of segment overlap. Defaults to 256.

    Returns:
        ndarray, axis=(time, ): f0 array
    """
    # zero padding at end of data
    padding_length = nperseg - len(data) % (nperseg - noverlap)
    data = np.concatenate([data, np.zeros(padding_length)])

    steps = (len(data) - noverlap) // (nperseg - noverlap)
    auto_correlation_ = np.zeros((nperseg, steps))
    peaks = np.zeros(steps)
    for step in range(steps):
        segment = data[
            step * (nperseg - noverlap): step * (nperseg - noverlap) + nperseg]
        
        # auto correlation
        auto_correlation_[:, step] = auto_correlation(segment)
        
        # find maximum peak
        maximum_candidates = np.zeros((0, 2))
        for i in range(nperseg - 2):
            if (auto_correlation_[i, step] <
                    auto_correlation_[i + 1, step] and
                auto_correlation_[i + 1, step] >
                    auto_correlation_[i + 2, step]):
                
                maximum_candidates = np.append(
                    maximum_candidates,
                    np.array([[(i + 1), auto_correlation_[i + 1, step]]]),
                    axis=0)
        if (len(maximum_candidates) != 0):
            maximum_candidates = \
                maximum_candidates[np.argsort(maximum_candidates[:, 1])[::-1]]
            peaks[step] = maximum_candidates[0, 0]
        else:
            # when no peaks found
            peaks[step] = nperseg

    return fs / peaks


def auto_correlation(data):
    """calculate auto correlation

    Args:
        data (ndarray, axis=(time, )): input data

    Returns:
        ndarray, axis=(frequency, ): auto correlation
    """

    length = len(data)
    result = np.zeros(length)
    for m in range(length):
        result[m] = np.sum(data[0: length - m] * data[m:])
    return result
