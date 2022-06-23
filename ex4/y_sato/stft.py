import librosa
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import soundfile as sf
import matplotlib.ticker


def stft(signal, window, step):
    Z = []
    win_fc=np.hamming(window)
    for i in range((signal.shape[0] - window) // step):
        tmp = signal[i*step : i*step + window]
        tmp = tmp * win_fc
        tmp = np.fft.fft(tmp)
        Z.append(tmp)
    Z = np.array(Z)
    return Z


def istft(y, frame_length, window, step):
    Z = np.zeros(frame_length)
    for i in range(len(y)) :
        tmp = np.fft.ifft(y[i])
        Z[i*step : i*step+window] += np.real(tmp)
    Z = np.array(Z)
    return Z


def convolution(input, filter):
    input_len = len(input)
    filter_len = len(filter)
    result = np.zeros(input_len + filter_len - 1)

    for i in range(input_len):
        result[i : i+filter_len] += np.multiply(input[i] , filter)

    return result
