import argparse

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import librosa
import librosa.display
import scipy.io.wavfile

from my_functions import stft
from my_functions import filter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ex2 LPF')
    parser.add_argument("-i", "--input", help="input file")
    parser.add_argument('--mode', choices=['LP', 'HP'], default='LP')
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
    Zxx, t, f = stft.stft(wav, sampling_rate)

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
    
    if (args.mode == "LP"):
        # LPF
        filter_ = filter.create_lpf(440)
        wav_filtered = filter.apply_filter(wav, filter_)

    elif (args.mode == "HP"):
        # HPF
        filter_ = filter.create_hpf(880)
        wav_filtered = filter.apply_filter(wav, filter_)

    # stft
    Zxx, t, f = stft.stft(wav_filtered, sampling_rate)

    # plot stft
    ax[1, 1].set_title(f"{args.mode} filtered spectrogram")
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
    
    # filter preview
    ax[2, 1].set_title(f"{args.mode} filter preview")
    ax[2, 1].plot(filter_)
    ax[2, 1].set_xlabel("sample [n]")
    ax[2, 1].set_ylabel("magnification")
    
    # filter property
    filter_property = np.fft.fft(filter_)
    ax[1, 0].set_title(f"{args.mode} filter property (amplitude)")
    ax[1, 0].plot(
        np.linspace(0, sampling_rate // 2, len(filter_) // 2),
        20 * np.log10(np.abs(filter_property[: len(filter_) // 2])))
    ax[1, 0].set_xlim(0, sampling_rate // 2)
    ax[1, 0].set_xlabel("Frequency [Hz]")
    ax[1, 0].set_ylabel("Amplitude [dB]")
    
    angle = np.angle(filter_property[: len(filter_) // 2])
    # np.place(angle, angle > -1 * 10 ** -3, -1 * np.pi)
    ax[2, 0].set_title(f"{args.mode} filter property (phase)")
    ax[2, 0].plot(
        np.linspace(0, sampling_rate // 2, len(filter_) // 2),
        angle)
    ax[2, 0].set_xlim(0, sampling_rate // 2)
    ax[2, 0].set_xlabel("Frequency [Hz]")
    ax[2, 0].set_ylabel("phase [rad]")

    # plt.savefig("result.png")
    plt.show()
    
    # export istft result as wav
    scipy.io.wavfile.write(f"{args.mode}.wav",
                           sampling_rate,
                           wav_filtered.astype(np.float32))
