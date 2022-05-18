import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os


def stft(y, hop=0.5, win_length=1024):
    """Compute the Short Time Fourier Transform (STFT).

    Args:
        y (np.ndarray, real-valued): Time series of measurement values.
        hop (float, optional): Hop (Overlap) size. Defaults to 0.5.
        win_length (int, optional): Window size. Defaults to 1024.

    Returns:
        np.ndarray: Complex-valued matrix of short-term Fourier transform coefficients.
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


def main():
    # load audio file
    # get current working directory
    dir = os.path.dirname(os.path.abspath(__file__)) + "/"
    # dir = "/work6/r_tanaka/venvs/py3venv/B4Lecture-2022/ex1/r_tanaka/"
    audio_path = dir + "ex1_sample.wav"
    # get waveform data and sample rate
    wav, sr = librosa.load(audio_path, mono=True)

    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
    plt.subplots_adjust(hspace=0.6)

    # draw original signal
    librosa.display.waveshow(wav, sr=sr, color="b", ax=ax[0])
    ax[0].set(title="Original signal", xlabel=None, ylabel="Magnitude")

    # parameter
    hop = 0.5
    win_length = 1024
    hop_length = int(win_length * hop)

    # STFT
    amp = stft(wav, hop=hop, win_length=win_length)
    # convert an amplitude spectrogram to dB_scaled spectrogram
    db = librosa.amplitude_to_db(np.abs(amp))
    # dB = librosa.amplitude_to_db(np.abs(librosa.stft(wav)), ref=np.max)
    # draw spectrogram (log scale)
    img = librosa.display.specshow(
        db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="log",
        ax=ax[1],
        cmap="plasma",
    )
    ax[1].set(title="Spectrogram", xlabel=None, ylabel="Frequency [Hz]")
    ax[1].set_yticks([0, 128, 512, 2048, 8192])
    fig.colorbar(img, aspect=10, pad=0.01, extend="both", ax=ax[1], format="%+2.f dB")

    # inverse-STFT
    inv_wav = istft(amp, hop=hop, win_length=win_length)
    # draw re-synthesized signal
    librosa.display.waveshow(inv_wav, sr=sr, color="b", ax=ax[2])
    ax[2].set(title="Re-synthesized signal", xlabel="Time [s]", ylabel="Magnitude")

    # graph adjustment
    ax_pos_0 = ax[0].get_position()
    ax_pos_1 = ax[1].get_position()
    ax_pos_2 = ax[2].get_position()
    ax[0].set_position([ax_pos_0.x0, ax_pos_0.y0, ax_pos_1.width, ax_pos_1.height])
    ax[2].set_position([ax_pos_2.x0, ax_pos_2.y0, ax_pos_1.width, ax_pos_1.height])
    # align the axis labels
    fig.align_labels()

    # save and show figure of result
    plt.savefig(dir + "ex1_result.png")
    plt.show()


if __name__ == "__main__":
    main()
