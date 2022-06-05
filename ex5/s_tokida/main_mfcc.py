import myfunc

import argparse
from tkinter import N
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.fftpack.realtransforms import dct

def cal_m0(f0):
    """calculate m0
    Args:
        f0 (float)
    Returns:
        m0 (float)
    """
    return 1000.0 / np.log(1000.0 / f0 + 1.0)

def hztomel(f, f0=700.0):
    """convert frequency [Hz] to mel frequency [mel]
    Args:
        f (ndarray): frequency [Hz]
    Returns:
        m (ndarray): mel frequency [mel]
    """
    m0 = cal_m0(f0)
    return m0 * np.log(f / f0 + 1.0)

def meltohz(m, f0=700.0):
    """convert mel frequency [mel] to frequency [Hz]
    Args:
        m (ndarray): mel frequency [mel]
    Returns:
        f (ndarray): frequency [Hz]
    """
    m0 = cal_m0(f0)
    return f0 * (np.exp(m / m0) -1.0)

def melfilterbank(sr, shift_size, n_channel):
    """make mel filter bank
    Args:
        sr (int): sample rate [Hz]
        shift_size (int): shift size
        n_channel (int): number of channel
    Returns:
        filterbank (ndarray): mel filter bank
    """

    fnyq = sr / 2  # nyquist frequency[Hz] = samplerate / 2
    melnyq = hztomel(fnyq)  # nyquist frequency[mel]
    nmax = shift_size // 2  # maximum number of frequency indexes
    df = sr / shift_size  # frequency resolution (Hz width per frequency index)

    dmel = melnyq / (n_channel + 1)  # center frequency of each filter on the Mel scale
    melcenters = np.arange(1, n_channel + 1) * dmel
    fcenters = meltohz(melcenters)
    indexcenter = np.round(fcenters / df)  # Convert center frequency to frequency index
    indexstart = np.hstack(([0], indexcenter[0:n_channel - 1]))
    indexstop = np.hstack((indexcenter[1:n_channel], [nmax]))

    filterbank = np.zeros((n_channel, nmax))

    for c in range(0, n_channel):
        # Find the point from the left of the triangular filter
        increment= 1.0 / (indexcenter[c] - indexstart[c])
        for i in range(int(indexstart[c]), int(indexcenter[c])):
            filterbank[c, i] = (i - indexstart[c]) * increment
        # Find the point from the right of the triangular filter
        decrement = 1.0 / (indexstop[c] - indexcenter[c])
        for i in range(int(indexcenter[c]), int(indexstop[c])):
            filterbank[c, i] = 1.0 - ((i - indexcenter[c]) * decrement)

    return filterbank


def calc_mfcc(data, shift_size, filterbank, dim):
    """get F0 by cepstrum

     Args:
         data (ndarray): input signal
         shift_size (int, optional): Length of window. Defaults to 1024.
         filterbank (ndarray): mel filter bank
         dim (int): number of dimention

     Returns:
         mfcc (ndarray): mfcc
     """

    spec = myfunc.stft(data, shift_size, data.shape[0], shift_size//2)
    spec = spec[:, : shift_size//2]
    # Apply mel filter bank
    mel_spec = 20 * np.log10(np.dot(np.abs(spec), filterbank.T))
    # Discrete cosine transform
    ceps = dct(mel_spec, type=2, norm = 'ortho', axis=-1)
    mfcc = ceps[:, :dim]  # defalt(dim = 12) 

    return mfcc

def calc_delta(mfcc, k=2):
    """calculate delta mfcc
    Args:
        mfcc (ndarray): mfcc
        k (int): number of frames of before and after
    Returns:
        delta mfcc (ndarray)
    """

    mfcc_pad = np.pad(mfcc, [(k, k+1), (0, 0)], 'edge')  # add frames
    k_sq = np.sum(np.arange(-k, k+1) ** 2)
    x = np.arange(-k, k+1)
    d_mfcc = np.zeros_like(mfcc)
    for i in range(mfcc.shape[0]):
        d_mfcc[i] = np.dot(x, mfcc_pad[i : i + k * 2 + 1])
    return d_mfcc / k_sq

def main():
    parser = argparse.ArgumentParser()

    # parser.add_argument
    parser.add_argument('filepath', type=str, help='wav file name : ex4.wav')
    parser.add_argument('--d', type=int, default=12, help='dimension of MFCC')

    args = parser.parse_args()
    filepath = args.filepath
    dim = args.d  # nceps

    # define numbers
    shift_size = 1024

    # load file
    data, samplerate = librosa.load(filepath)  # samplerate: 48000
    time = float(data.shape[0] / samplerate)

    # make melfilterbank
    n_channel = 20
    df = samplerate / shift_size
    filterbank = melfilterbank(samplerate, shift_size, n_channel)

    plt.figure(figsize=(8,6))
    for c in np.arange(0, n_channel):
        plt.plot(np.arange(0, shift_size/2) * df, filterbank[c])
    plt.title('MelFilterBank')
    plt.xlabel('Frequency[Hz]')
    plt.savefig('fig/melfilterbank.png')
    plt.show()
    plt.close()
    
    # mfcc
    mfcc = calc_mfcc(data, shift_size, filterbank, dim)

    # Δmfcc
    d_mfcc = calc_delta(mfcc)

    # ΔΔmfcc
    dd_mfcc = calc_delta(d_mfcc)

    # plot spectrogram
    fig = plt.figure(figsize=(8,8))

    ax = fig.add_subplot(411)
    myfunc.myspectrogram(data, shift_size, data.shape[0], shift_size//2, samplerate, 'Spectrogram', 'rainbow')

    ax1 = fig.add_subplot(412)
    ax1.set(ylabel='[mel]', title='MFCC')
    im = ax1.imshow(mfcc.T, extent=[0, time, 0, dim], aspect='auto', origin='lower', cmap= 'rainbow')
    cbar = fig.colorbar(im)

    ax2 = fig.add_subplot(413)
    ax2.set(ylabel='[mel]', title='ΔMFCC')
    im = ax2.imshow(d_mfcc.T, extent=[0, time, 0, dim], aspect='auto', origin='lower', cmap='rainbow')
    cbar = fig.colorbar(im)

    ax3 = fig.add_subplot(414)
    ax3.set(xlabel='Time[s]', ylabel='[mel]', title='ΔΔMFCC')
    im = ax3.imshow(dd_mfcc.T, extent=[0, time, 0, dim], aspect='auto', origin='lower', cmap='rainbow')
    cbar = fig.colorbar(im)

    plt.tight_layout()
    plt.savefig('fig/spectrogram.png')
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()