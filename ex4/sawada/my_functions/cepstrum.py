import numpy as np


def cepstrum(data):
    """calcurate cepstrum

    Args:
        data (ndarray, axis=(time)): input data

    Returns:
        ndarray: cepstrum
    """
    data_length = len(data)
    
    # multiplying hanning window
    windowed_data = data * np.hanning(data_length)
    
    # fft
    fft_data = np.fft.fft(windowed_data)
    
    # log amplitude
    log_data = np.log10(np.abs(fft_data))
    
    # cepstrum
    return np.real(np.fft.ifft(log_data))


def craete_lifter(length, mode="lp", cutoff_frame=20):
    """create lifter

    Args:
        length (int): lifter length
        mode (str): "hp" or "lp". Defaults to "lp"
        cutoff_frame (int, optional): cutoff cepstrum coefficient.
            Defaults to 20.

    Returns:
        ndarray: lifter
    """
    lifter = np.zeros(length)
    lifter[0: cutoff_frame] = 1
    # if hp lifter, reverse 1 and 0
    if (mode == "hp"):
        lifter = 1 - lifter
    return lifter


def st_cepstrum(data, fs, nperseg=512, noverlap=256):
    """short-time cepstrum analysis

    Args:
        data (ndarray, axis=(time, )): input data
        fs (int): sampling rate
        nperseg (int, optional): samples in st cepstrum segment.
            Defaults to 512.
        noverlap (int, optional): samples of st cepstrum overlap.
            Defaults to 256.

    Returns:
        ndarray, axis=(time, quefrency): short-time cepstrum
    """
    # zero padding at end of data
    padding_length = nperseg - len(data) % (nperseg - noverlap)
    data = np.concatenate([data, np.zeros(padding_length)])
    
    # shot-time cepstrum analysis
    steps = (len(data) - noverlap) // (nperseg - noverlap)
    cepstrums = np.zeros((steps, nperseg))
    for i in range(steps):
        cepstrums[i] = cepstrum(
            data[(nperseg - noverlap) * i: (nperseg - noverlap) * i + nperseg])
    
    return cepstrums
