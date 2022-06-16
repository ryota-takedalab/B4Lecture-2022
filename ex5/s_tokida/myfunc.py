import numpy as np
from matplotlib import pyplot as plt
from math import pi


def stft(data, shift_size, data_size, overlap): 
    """Short Time Fourier Transform

     Args:
         data (ndarray): input signal
         shift_size (int, optional): Length of window. Defaults to 1024.
         data_size (int): Length of data.]
         overlap (int, optional): Length of overlap. Defaults to 512.

     Returns:
         ndarray: spectrogram
     """

    shift = int((data_size-overlap)//(shift_size-overlap))  # number of shifts
    win = np.hamming(shift_size)  # humming window fuction
    spec = []
    for i in range(shift):
        shift_data = data[i*(shift_size-overlap):i*(shift_size-overlap)+shift_size]
        spec.append(np.fft.fft(win * shift_data))  # fft

    return np.array(spec)

def myspectrogram(data, shift_size, data_size, overlap, samplerate, title_name, cmap):
    """Plot spectrogram

     Args:
         data (ndarray): input signal
         shift_size (int, optional): Length of window. Defaults to 1024.
         data_size (int): Length of data.]
         overlap (int, optional): Length of overlap. Defaults to 512.
         samplerate (int): samplerate
         title_name (str): title name
         cmap (str): cmap
     """
    
    spec = stft(data, shift_size, data_size, overlap)
    spec_log = 20*np.log10(np.abs(spec))
    plt.imshow(spec_log[:, :shift_size//2].T, extent=[0, float(data_size)/samplerate, 0, samplerate],
               aspect='auto', origin = 'lower', cmap = cmap)  # extent 表示する場所を明示
    plt.colorbar()
    plt.ylabel('Frequency[Hz]')
    plt.title(title_name)