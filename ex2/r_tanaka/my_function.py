import numpy as np
from math import pi


def stft(y, hop=0.5, win_length=1024):
    """Compute the Short Time Fourier Transform (STFT).

    Args:
        y (np.ndarray, real-valued): Time series of measurement values.
        hop (float, optional): Hop (Overlap) size. Defaults to 0.5.
        win_length (int, optional): Window size. Defaults to 1024.

    Returns:
        np.ndarray: Complex-valued matrix of
        short-term Fourier transform coefficients.
    """
    hop_length = int(win_length * hop)
    # Number of row in array y
    ynum = y.shape[0]
    # prepare a hamming window
    window = np.hamming(win_length)

    F = []
    for i in range(int((ynum - hop_length) / hop_length)):
        # extract the part of array y to which the FFT is applied
        tmp = y[i * hop_length: i * hop_length + win_length]
        # multiplied by window function
        tmp = tmp * window
        # Fast Fourier Transform (FFT)
        tmp = np.fft.rfft(tmp)
        # add tmp to the end of array F
        F.append(tmp)

    # (frame, freq) -> (freq, frame)
    F = np.transpose(F)
    return F


def istft(F, hop=0.5, win_length=1024):
    """Compute the Inverse Short Time Fourier Transform (ISTFT).

    Args:
        F (np.ndarray): Complex-valued matrix of short-term Fourier transform coefficients.
        hop (float, optional): Hop (Overlap) size. Defaults to 0.5.
        win_length (int, optional): Window size. Defaults to 1024.

    Returns:
        np.ndarray: Time domain signal.
    """

    hop_length = int(win_length * hop)
    # prepare a hamming window
    window = np.hamming(win_length)
    # (freq, frame) -> (frame, freq)
    F = np.transpose(F)
    # Inverse Fast Fourier Transform (IFFT)
    tmp = np.fft.irfft(F)
    # divided by window function
    tmp = tmp / window
    # remove overlap
    tmp = tmp[:, :hop_length]
    y = tmp.reshape(-1)

    return y


def convolve(original_array, filter_array):
    """compute the convolution operation.

    Args:
        original_array (array_like): First one-dimensional input array.
        filter_array (array_like): Second one-dimensional input array.

    Returns:
        ndarray: convolution of original_array and filter_array.
    """
    # get the number of elements of the array
    original_num = len(original_array)
    filter_num = len(filter_array)
    # prepare the array after processing
    processed_array = np.zeros(original_num + filter_num - 1)

    # compute the convolution operation
    for i in range(filter_num):
        processed_array[i: i + original_num] += np.multiply(original_array, filter_array[i])

    return processed_array


def sinc(x):
    """sinc function

    Args:
        x (float): input.

    Returns:
        float: apply sinc function.
    """
    if x == 0:
        return 1.0
    else:
        return np.sin(x)/x


def LPF_impulse(n, f_low, sr):
    # cutoff angular frequency
    w_low = 2 * pi * f_low
    # normalized angular freqency
    wn_low = w_low / sr
    return wn_low * sinc(wn_low * n) / pi


def LowPassFilter(f_low, sr, N=30):
    """make low pass filter

    Args:
        f_low (float): low edge frequency
        sr (float): sampling rate
        N (int): the number of filter coefficients. Defaults to 30.

    Returns:
        ndarray: filter coefficient
    """
    # if N is an even number, change N to an odd number.
    if N % 2 != 0:
        N += 1
    N = int(N)

    tmp = []
    # calculate filter coefficients
    for n in range(-N // 2, N // 2 + 1):
        tmp.append(LPF_impulse(n, f_low, sr))

    # Apply window function method
    tmp = np.array(tmp)
    window_func = np.hamming(N + 1)
    return tmp * window_func
