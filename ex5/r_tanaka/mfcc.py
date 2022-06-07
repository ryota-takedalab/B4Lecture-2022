import numpy as np
import os
import librosa
import librosa.display
import scipy
import matplotlib.pyplot as plt


def preEmphasis(x, p=0.97):
    """apply pre-emphasis filter

    Args:
        x (np.ndarray): input data.
        p (float): filter coefficience. Default to 0.97.

    Returns:
        np.ndarray: pre-emphasis filter.
    """
    # apply pre-emphasis filter
    return scipy.signal.lfilter([1.0, -p], 1, x)


def hz2mel(f):
    """convert Hz to mel

    Args:
        f (np.ndarray): input data (Hz)

    Returns:
        np.ndarray: output data (mel)
    """
    return 2595 * np.log(f / 700.0 + 1.0)


def mel2hz(m):
    """convert mel to hz

    Args:
        m (np.ndarray): input data (mel)

    Returns:
        np.ndarray: output data (Hz)
    """
    return 700 * (np.exp(m / 2595) - 1.0)


def melFilterBank(sr, N, numChannels):
    """Create mel-filter bank

    Args:
        sr (int): sampling rate
        N (int): window size
        numChannels (int): number of channels in mel-filter bank

    Returns:
        np.ndarray: mel-filter bank
    """
    # nyquist frequency（Hz）
    fmax = sr / 2
    # nyquist frequency（mel）
    melmax = hz2mel(fmax)
    # maximum number of frequency indexes
    nmax = N // 2
    # frequency resolution (Hz width per frequency index 1)
    df = sr / N
    # find the center frequency of each filter in the Mel scale
    dmel = melmax / (numChannels + 1)
    melcenters = np.arange(1, numChannels + 1) * dmel
    # convert the center frequency of each filter to Hz
    fcenters = mel2hz(melcenters)
    # convert the center frequency of each filter to a frequency index
    indexcenter = np.round(fcenters / df)
    # index of the start position of each filter
    indexstart = np.hstack(([0], indexcenter[0:numChannels - 1]))
    # index of the end position of each filter
    indexstop = np.hstack((indexcenter[1:numChannels], [nmax]))
    filterbank = np.zeros((numChannels, nmax))
    for c in range(0, numChannels):
        # find points from the slope of the light line of the triangular filter
        increment = 1.0 / (indexcenter[c] - indexstart[c])
        for i in range(int(indexstart[c]), int(indexcenter[c])):
            filterbank[c, i] = (i - indexstart[c]) * increment
        # find points from the slope of the right line of the triangular filter
        decrement = 1.0 / (indexstop[c] - indexcenter[c])
        for i in range(int(indexcenter[c]), int(indexstop[c])):
            filterbank[c, i] = 1.0 - ((i - indexcenter[c]) * decrement)

    return filterbank


def get_mfcc(x, sr, win_length=1024, hop_length=512, mfcc_dim=12):
    """get mfcc

    Args:
        x (np.ndarray): input data
        sr (int): sampling rate
        win_length (int, optional): window size. Defaults to 1024.
        hop_length (int, optional): hop length. Defaults to 512.
        mfcc_dim (int, optional): dimension number of mfcc. Defaults to 12.

    Returns:
        np.ndarray: mfcc
    """
    # Number of row in array x
    xnum = x.shape[0]
    # prepare a hamming window
    window = np.hamming(win_length)

    mfcc = []
    for i in range(int((xnum - hop_length) / hop_length)):
        # extract the part of array x to calcurate mfcc
        tmp = x[i * hop_length: i * hop_length + win_length]
        # multiplied by window function
        tmp = tmp * window
        # Fast Fourier Transform (FFT)
        tmp = np.fft.rfft(tmp)
        # convert to absolute values to obtain the power spectrum
        tmp = np.abs(tmp)
        tmp = tmp[:win_length//2]
        # get mel-filter bank
        # number of channels in mel-filter bank
        numChannels = 26
        filterbank = melFilterBank(sr, win_length, numChannels)
        # get mel-scale spectrogram
        tmp = np.dot(filterbank, tmp)
        # logarithmize.
        tmp = 20 * np.log10(tmp)
        # Discrete Cosine Transform (DCT) to obtain the cepstrum
        tmp = scipy.fftpack.dct(tmp, norm='ortho')
        # apply lifter to get mfcc
        tmp = tmp[1:mfcc_dim+1]

        mfcc.append(tmp)

    # (frame, freq) -> (freq, frame)
    mfcc = np.transpose(mfcc)
    return mfcc


def delta(input, mfcc_dim, k):
    """get Δmfcc (or ΔΔmfcc).

    Args:
        input (np.ndarray): mfcc (or Δmfcc)
        mfcc_dim (int): dimension number of mfcc (or Δmfcc)
        k (int): number of frames to get Δmfcc (or ΔΔmfcc)

    Returns:
        np.ndarray: Δmfcc (or ΔΔmfcc)
    """
    output = np.zeros((mfcc_dim, input.shape[1] - 2 * k))
    for i in range(input.shape[1] - 2 * k):
        # extract k frames each from the front and back
        tmp = input[:, i: i + 2 * k + 1]
        delta = [0] * mfcc_dim
        # calculate the amount of variation for each dimension
        for j in range(mfcc_dim):
            a, _ = np.polyfit(np.arange(-k, k + 1, 1), tmp[j:j + 1, :][0], 1)
            delta[j] = a
        delta = np.array(delta).reshape(mfcc_dim, -1)
        output[:, i:i + 1] = delta

    return output


def main():
    # load audio file
    # get current working directory
    dir = os.path.dirname(os.path.abspath(__file__)) + "/"
    audio_path = dir + "sample.wav"
    # get waveform data and sample rate
    wav, sr = librosa.load(audio_path, mono=True)

    # set drawing area
    plt.rcParams["figure.figsize"] = (14, 11)
    fig, ax = plt.subplots(4, 1)
    fig.tight_layout(rect=[0.05, 0, 1, 0.95])
    fig.subplots_adjust(hspace=0.3)

    # set paramaters
    win_length = 512
    hop_length = 256

    # get original spectrogram
    spectrogram = librosa.stft(wav, win_length=win_length, hop_length=hop_length)
    spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram))

    # plot original spectrogram on ax[0]
    img = librosa.display.specshow(spectrogram_db, y_axis="log", sr=sr, cmap="rainbow", ax=ax[0])
    ax[0].set(title="original spectrogram", xlabel="time[s]", ylabel="frequency [Hz]")
    fig.colorbar(img, aspect=10, pad=0.01, extend="both", ax=ax[0], format="%+2.f dB")

    # apply pre-emphasis filter
    wav = preEmphasis(wav)

    # mfcc
    mfcc_dim = 12
    mfcc = get_mfcc(wav, sr, win_length=win_length, hop_length=hop_length, mfcc_dim=mfcc_dim)
    wav_time = wav.shape[0] // sr
    extent = [0, wav_time, 0, mfcc_dim]
    img1 = ax[1].imshow(np.flipud(mfcc), aspect="auto", extent=extent, cmap="rainbow")
    ax[1].set(title="MFCC sequence", xlabel="time[s]", ylabel="MFCC", yticks=range(0, 13, 2))
    fig.colorbar(img1, aspect=10, pad=0.01, extend="both", ax=ax[1], format="%+2.f dB")

    # Δmfcc
    dmfcc = delta(mfcc, mfcc_dim, k=2)
    img2 = ax[2].imshow(np.flipud(dmfcc), aspect="auto", extent=extent, cmap="rainbow")
    ax[2].set(title="ΔMFCC sequence", xlabel="time[s]", ylabel="ΔMFCC", yticks=range(0, 13, 2))
    fig.colorbar(img2, aspect=10, pad=0.01, extend="both", ax=ax[2], format="%+2.f dB")

    # ΔΔmfcc
    ddmfcc = delta(dmfcc, mfcc_dim, k=2)
    img3 = ax[3].imshow(np.flipud(ddmfcc), aspect="auto", extent=extent, cmap="rainbow")
    ax[3].set(title="ΔΔMFCC sequence", xlabel="time[s]", ylabel="ΔΔMFCC", yticks=range(0, 13, 2))
    fig.colorbar(img3, aspect=10, pad=0.01, extend="both", ax=ax[3], format="%+2.f dB")

    fig.tight_layout()
    fig.savefig("mfcc.png")


if __name__ == "__main__":
    main()
