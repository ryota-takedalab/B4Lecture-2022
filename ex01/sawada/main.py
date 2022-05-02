import numpy as np
import matplotlib.pyplot as plt
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
    # read audio file
    filename = "input.wav"
    sampling_rate = 16000
    wav, _ = librosa.load(filename, sr=sampling_rate, mono=True)

    # preview audio
    t = np.linspace(0, len(wav) / sampling_rate, len(wav))
    plt.title("input audio")
    plt.plot(t, wav)
    plt.xlabel("time(s)")
    plt.ylabel("amplitude")
    plt.show()
    # plt.savefig("preview.png")
    plt.close()

    # stft
    Zxx, t, f = stft(wav, sampling_rate)

    # plot stft
    plt.title("stft")
    librosa.display.specshow(np.abs(Zxx) ** 2,
                             sr=sampling_rate * 2,
                             x_axis="s",
                             y_axis="linear")
    plt.colorbar()
    plt.show()
    # plt.savefig("stft.png")
    plt.close()

    # istft
    reconstructed = istft(Zxx)

    # export istft result as wav
    scipy.io.wavfile.write("reconstructed.wav", sampling_rate, reconstructed)

    # plot istft
    t = np.linspace(0, len(reconstructed) / sampling_rate, len(reconstructed))
    plt.title("istft")
    plt.plot(t, reconstructed)
    plt.xlabel("time(s)")
    plt.ylabel("amplitude")
    plt.show()
    # plt.savefig("reconstructed.png")
    plt.close()
