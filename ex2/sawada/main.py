import argparse

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import librosa
import librosa.display
import scipy.io.wavfile


def _create_base_filter(cutoff_frequency, nperseg=512, sr=16000):
    """filter creation base function
    
    This function returns the basic LPF for creating filter, wrapped and used.

    Args:
        cutoff_frequency (int): cutoff frequency
        nperseg (int, optional): samples in stft segment. Defaults to 512.
        sr (int, optional): sampling rate. Defaults to 16000.
        
    Returns:
        ndarray: filter
    """
    
    regularized_cutoff_frequency = 2 * np.pi * cutoff_frequency / sr
    t = np.arange(nperseg)
    
    # create filter
    filter = 2 * regularized_cutoff_frequency \
        * np.sinc(regularized_cutoff_frequency / np.pi * (t - nperseg // 2)) \
        * np.hamming(nperseg)
    return filter


def create_lpf(cutoff_frequency, fft_size=512, sr=16000):
    """low-pass filter

    Args:
        cutoff_frequency (int): cutoff frequency
        fft_size (int, optional): samples in stft segment. Defaults to 512.
        sr (int, optional): sampling rate. Defaults to 16000.

    Returns:
        ndarray: low-pass filter
    """
    return _create_base_filter(cutoff_frequency, fft_size, sr)


def create_hpf(cutoff_frequency, fft_size=512, sr=16000):
    """high-pass filter

    Args:
        cutoff_frequency (int): cutoff frequency
        fft_size (int, optional): samples in stft segment. Defaults to 512.
        sr (int, optional): sampling rate. Defaults to 16000.

    Returns:
        ndarray: high-pass filter
    """
    return _create_base_filter(sr / (2 * np.pi)) \
        - _create_base_filter(cutoff_frequency, fft_size, sr)


def apply_filter(audio, filter):
    """apply filter to audio

    Args:
        audio (ndarray): audio array
        filter (ndarray): filter

    Returns:
        ndarray: filtered audio
    """
    return np.convolve(audio, filter)


def stft(audio, fs, nperseg=512, noverlap=256):
    """Short-Time Fourier Transform

    Args:
        audio (array_like): input audio array
        fs (float): sampling rate of input audio
        nperseg (int, optional): samples in stft segment. Defaults to 512.
        noverlap (int, optional): samples of stft overlap. Defaults to 256.

    Returns:
        ndarray: stft of audio
        ndarray: time axis
        ndarray: frequency axis
    """

    # zero padding at end of audio
    padding_length = nperseg - len(audio) % (nperseg - noverlap)
    audio = np.concatenate([audio, np.zeros(padding_length)])

    # create window (hanning)
    window = np.hanning(nperseg)

    # stft
    steps = (len(audio) - noverlap) // (nperseg - noverlap)
    Zxx = np.empty((0, nperseg))
    for i in range(steps):
        # apply window
        sample = window * audio[
            (nperseg - noverlap) * i: (nperseg - noverlap) * i + nperseg
        ]
        # fft
        Zxx = np.append(Zxx, [np.fft.fft(sample)], axis=0)

    # create time axis
    t = np.linspace(0, len(audio) / fs, Zxx.shape[0])

    # create frequency axis
    f = np.linspace(0, fs, Zxx.shape[1])

    return Zxx.T, t, f


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ex2 LPF')
    parser.add_argument("-i", "--input", help="input file")
    args = parser.parse_args()

    # read audio file
    filename = args.input
    sampling_rate = 16000
    wav, _ = librosa.load(filename, sr=sampling_rate, mono=True)
    
    # plot initialization
    fig, ax = plt.subplots(3, 2, figsize=(12, 10))
    fig.subplots_adjust(hspace=0.6)
    
    # preview audio
    t = np.linspace(0, len(wav) / sampling_rate, len(wav))
    ax[0, 0].set_title("input audio")
    ax[0, 0].plot(t, wav)
    ax[0, 0].set_xlabel("Time [s]")
    ax[0, 0].set_ylabel("Amplitude")
    ax[0, 0].set_xlim([t[0], t[-1]])
    
    # stft
    Zxx, t, f = stft(wav, sampling_rate)

    # plot stft
    ax[0, 1].set_title("spectrogram")
    im = ax[0, 1].imshow(
        20 * np.log10(np.abs(np.flipud(Zxx[:Zxx.shape[0] // 2]))),
        cmap=plt.cm.jet,
        aspect="auto",
        extent=[t[0], t[-1], f[0], f[len(f) // 2]])
    ax[0, 1].set_xlabel("Time [s]")
    ax[0, 1].set_ylabel("Frequency [Hz]")
    # colorbar: https://sabopy.com/py/matplotlib-18/
    divider = make_axes_locatable(ax[0, 1])
    cax = divider.append_axes("bottom", size="5%", pad=0.5)
    plt.colorbar(im, cax=cax, orientation="horizontal")
    
    # LPF
    low_pass_filter = create_lpf(440)
    wav_lpf = apply_filter(wav, low_pass_filter)

    # stft
    Zxx, t, f = stft(wav_lpf, sampling_rate)

    # plot stft
    ax[1, 1].set_title("low-pass spectrogram")
    im = ax[1, 1].imshow(
        20 * np.log10(np.abs(np.flipud(Zxx[:Zxx.shape[0] // 2]))),
        cmap=plt.cm.jet,
        aspect="auto",
        extent=[t[0], t[-1], f[0], f[len(f) // 2]])
    ax[1, 1].set_xlabel("Time [s]")
    ax[1, 1].set_ylabel("Frequency [Hz]")
    # colorbar: https://sabopy.com/py/matplotlib-18/
    divider = make_axes_locatable(ax[1, 1])
    cax = divider.append_axes("bottom", size="5%", pad=0.5)
    plt.colorbar(im, cax=cax, orientation="horizontal")

    # HPF
    high_pass_filter = create_hpf(440)
    wav_hpf = apply_filter(wav, high_pass_filter)

    # stft
    Zxx, t, f = stft(wav_hpf, sampling_rate)

    # plot stft
    ax[2, 1].set_title("high-pass spectrogram")
    im = ax[2, 1].imshow(
        20 * np.log10(np.abs(np.flipud(Zxx[:Zxx.shape[0] // 2]))),
        cmap=plt.cm.jet,
        aspect="auto",
        extent=[t[0], t[-1], f[0], f[len(f) // 2]])
    ax[2, 1].set_xlabel("Time [s]")
    ax[2, 1].set_ylabel("Frequency [Hz]")
    # colorbar: https://sabopy.com/py/matplotlib-18/
    divider = make_axes_locatable(ax[2, 1])
    cax = divider.append_axes("bottom", size="5%", pad=0.5)
    plt.colorbar(im, cax=cax, orientation="horizontal")
    
    # filter property
    filter_property = np.fft.fft(low_pass_filter)
    ax[1, 0].set_title("low-pass filter property (amplitude)")
    ax[1, 0].plot(
        np.linspace(0, sampling_rate // 2, len(low_pass_filter) // 2),
        20 * np.log10(np.abs(filter_property[: len(low_pass_filter) // 2])))
    ax[1, 0].set_xlim(0, sampling_rate // 2)
    ax[1, 0].set_xlabel("Frequency [Hz]")
    ax[1, 0].set_ylabel("Amplitude [dB]")
    
    angle = np.angle(filter_property[: len(low_pass_filter) // 2])
    # np.place(angle, angle > -1 * 10 ** -3, -1 * np.pi)
    ax[2, 0].set_title("low-pass filter property (phase)")
    ax[2, 0].plot(
        np.linspace(0, sampling_rate // 2, len(low_pass_filter) // 2),
        angle)
    ax[2, 0].set_xlim(0, sampling_rate // 2)
    ax[2, 0].set_xlabel("Frequency [Hz]")
    ax[2, 0].set_ylabel("phase [rad]")

    # plt.savefig("result.png")
    plt.show()
    
    # export istft result as wav
    scipy.io.wavfile.write("low-pass.wav",
                           sampling_rate,
                           wav_lpf.astype(np.float32))
