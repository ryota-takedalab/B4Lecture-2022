import numpy as np
from matplotlib import pyplot as plt
from math import pi


def stft(data, shift_size, data_size, overlap): 

    shift = int((data_size-overlap)//(shift_size-overlap))  # number of shifts
    win = np.hamming(shift_size)  # humming window fuction
    # spec = np.zeros([shift, shift_size], dtype = np.complex)
    spec = []
    for i in range(shift):
        shift_data = data[int(i*(shift_size-overlap)):int(i*(shift_size-overlap)+shift_size)]
        spec.append(np.fft.fft(win * shift_data))  # fft

    return spec

def myspectrogram(data, shift_size, data_size, overlap, samplerate, title_name, cmap):
    spec = stft(data, shift_size, data_size, overlap)
    spec_log = 20*np.log10(np.abs(spec))
    plt.imshow(spec_log[:, :shift_size//2].T, extent=[0, float(data_size)/samplerate, 0, samplerate],
               aspect='auto', origin = 'lower', cmap = cmap)
    plt.colorbar()

def myconvolve(x, h):
    y = np.zeros(len(x) + len(h))
    for i in range(len(x)):
        y[i: i+len(h)] += x[i] * h
    return y[:len(x)]

def sinc(x):
    if x == 0:
        return 1.0
    else:
        return np.sin(x) / x

def befilter(f_low, f_high, samplerate, f_size):
    if f_size % 2 != 0:
        f_size += 1
    
    w_low = 2 * pi * f_low / samplerate
    w_high = 2 * pi * f_high / samplerate

    fir = []
    for n in range(-f_size // 2, f_size // 2 + 1):
        BEF_impulse = sinc(pi * n) + (- w_high * sinc(w_high * n) + w_low * sinc(w_low * n)) / pi
        fir.append(BEF_impulse)

    fir = np.array(fir)
    window = np.hamming(f_size + 1)

    return fir * window
