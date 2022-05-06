import argparse

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import librosa
import librosa.display
import scipy.io.wavfile


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


def istft(Zxx, nperseg=512, noverlap=256):
    """Inverse Short-Time Fourier Transform

    Args:
        Zxx (ndarray): spectrogram
        nperseg (int, optional): samples in stft segment. Defaults to 512.
        noverlap (int, optional): samples of stft overlap. Defaults to 256.

    Returns:
        ndarray: reconstructed audio
    """

    # initialize audio array with zeros
    audio = np.zeros((nperseg - noverlap) * (Zxx.shape[1] - 1) + nperseg)

    # istft
    for i, s in enumerate(Zxx.T):
        audio[
            i * (nperseg - noverlap): i * (nperseg - noverlap) + nperseg
        ] += np.fft.ifft(s).real
    return audio


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ex01 stft')
    parser.add_argument("-i", "--input", help="input file")
    args = parser.parse_args()

    # read audio file
    filename = args.input
    sampling_rate = 16000
    wav, _ = librosa.load(filename, sr=sampling_rate, mono=True)
    
    # plot initialization
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.45)

    # preview audio
    t = np.linspace(0, len(wav) / sampling_rate, len(wav))
    ax[0].set_title("input audio")
    ax[0].plot(t, wav)
    ax[0].set_xlabel("Time(s)")
    ax[0].set_ylabel("Amplitude")

    # stft
    Zxx, t, f = stft(wav, sampling_rate)

    # plot stft
    ax[1].set_title("spectrogram")
    im = ax[1].imshow(20 * np.log10(np.abs(Zxx)), cmap=plt.cm.jet,
                      aspect="auto", extent=[t[0], t[-1], f[0], f[-1]])
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Frequency (Hz)")
    # colorbar: https://sabopy.com/py/matplotlib-18/
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("bottom", size="5%", pad=0.5)
    plt.colorbar(im, cax=cax, orientation="horizontal")

    # istft
    reconstructed = istft(Zxx)
    # reconstructed = istft(np.abs(Zxx))

    # export istft result as wav
    scipy.io.wavfile.write("reconstructed.wav",
                           sampling_rate,
                           reconstructed.astype(np.float32))

    # plot istft
    t = np.linspace(0, len(reconstructed) / sampling_rate, len(reconstructed))
    ax[2].set_title("istft reconstructed audio")
    ax[2].plot(t, reconstructed)
    ax[2].set_xlabel("Time (s)")
    ax[2].set_ylabel("Amplitude")
    # plt.show()
    plt.savefig("result.png")
    plt.close()
