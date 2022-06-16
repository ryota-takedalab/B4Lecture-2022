import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.fftpack.realtransforms import dct


def stft(data, window=1024, step=512):
    """Short Time Fourier Transform

    Args:
        data (ndarray): input signal
        window (int, optional): Length of window. Defaults to 1024.
        step (int, optional): Length of shift. Defaults to 512.

    Returns:
        ndarray: spectrogram
    """
    # window function
    win_fc = np.hamming(window)

    frame = (len(data) - window + step) // step

    framed_signal = np.lib.stride_tricks.as_strided(
        data, shape=(window, frame), strides=(data.strides[0], data.strides[0] * step)
    )

    # win_fc.shape=(window,) -> win_fc[:,np.newaxis].shape=(window,1)
    window_signal = framed_signal * win_fc[:, np.newaxis]
    # fast fourier transform
    spec = np.fft.rfft(window_signal, axis=0)
    # -----------

    return spec


class MFCC:
    def __init__(self, data, sr, win, step, n_channels, f0) -> None:
        self.data = data
        self.sr = sr
        self.win = win
        self.step = step
        self.n_channels = n_channels
        self.f0 = f0
        # frequency resolution
        self.df = self.sr / self.win
        # make mel filter bank
        self.filterbank_ = self.melFilterBank()

    def freq_to_mel(self, f):
        """Convert frequency to mel frequency

        Args:
            f (ndarray): frequency

        Returns:
            ndarray: mel frequency
        """
        m0 = 1000 / np.log(1000 / self.f0 + 1)
        return m0 * np.log(f / self.f0 + 1)

    def mel_to_freq(self, m):
        """Convert mel frequency to frequency

        Args:
            m (ndarray): mel frequency

        Returns:
            ndarray: frequency
        """
        m0 = 1000 / np.log(1000 / self.f0 + 1)
        return self.f0 * (np.exp(m / m0) - 1)

    def melFilterBank(self):
        """Make mel filter bank

        Returns:
            ndarray: mel filter bank
        """
        # nyquist frequency
        f_nyq = self.sr / 2
        m_nyq = self.freq_to_mel(f_nyq)
        # maximum frequency index
        nmax = self.win // 2

        # calculate center frequencies in mel-scaled in each filter
        dmel = m_nyq / (self.n_channels + 1)
        m_centers = np.arange(1, self.n_channels + 1) * dmel
        # convert the center frequency to Hz-scale in each filter
        f_centers = self.mel_to_freq(m_centers)
        # convert the center frequency to frequency index in each filter
        idx_center = np.round(f_centers / self.df)
        # index of start position of each filter
        idx_start = np.hstack(([0], idx_center[0 : self.n_channels - 1]))
        # index of end position of each filter
        idx_end = np.hstack((idx_center[1 : self.n_channels], [nmax]))
        filterbank = np.zeros((self.n_channels, nmax))

        for i in range(0, self.n_channels):
            # calculate points from slope of left
            increment = 1.0 / (idx_center[i] - idx_start[i])
            for j in range(int(idx_start[i]), int(idx_center[i])):
                filterbank[i, j] = (j - idx_start[i]) * increment

            # calculate points from slope of right
            decrement = 1.0 / (idx_end[i] - idx_center[i])
            for j in range(int(idx_center[i]), int(idx_end[i])):
                filterbank[i, j] = 1.0 - ((j - idx_center[i]) * decrement)

        return filterbank

    def calc_mfcc(self):
        """Calculate MFCC

        Returns:
            ndarray, ndarray: mel scale spectrogram, MFCC
        """
        spec = stft(self.data)
        mel_spec = np.dot(self.filterbank_, np.abs(spec[:-1]))

        mfcc = np.zeros_like(mel_spec)
        for i in range(mel_spec.shape[1]):
            mfcc[:, i] = dct(mel_spec[:, i], type=2, norm="ortho", axis=-1)

        return mel_spec, mfcc

    def delta_mfcc(self, mfcc, k=2):
        """Calculate delta of MFCC

        Args:
            mfcc (ndarray): MFCC
            k (int, optional): window of regression. Defaults to 2.

        Returns:
            ndarray: delta of MFCC
        """
        mfcc_pad = np.pad(mfcc, [(0, 0), (k, k + 1)], "edge")
        k_sq = np.sum(np.arange(-k, k + 1) ** 2)
        m = np.arange(-k, k + 1)
        d_mfcc = np.zeros_like(mfcc)
        for i in range(mfcc.shape[1]):
            d_mfcc[:, i] = np.dot(mfcc_pad[:, i : i + k * 2 + 1], m.T)
        return d_mfcc / k_sq

    def mfcc_plot(self, filename):
        # plot mel filter bank
        for i in range(self.n_channels):
            plt.plot(np.arange(0, self.win // 2) * self.df, self.filterbank_[i])

        plt.title("Mel Filter Bank")
        plt.xlabel("Frequency[Hz]")
        plt.savefig("MelFilterBank.png")
        plt.show()

        fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(10, 6))
        plt.subplots_adjust(hspace=0.6)

        mel_spec_, mfcc_ = self.calc_mfcc()

        # mel spectrogram
        time = self.data.shape[0] // self.sr
        f_nyq = self.sr // 2
        extent = [0, time, 0, f_nyq]

        img = ax[0].imshow(
            librosa.amplitude_to_db(mel_spec_),
            aspect="auto",
            extent=extent,
            cmap="rainbow",
        )
        ax[0].set(
            title="Mel Spectrogram",
            xlabel=None,
            ylabel="Mel frequency [mel]",
            ylim=[0, 8000],
            yticks=range(0, 10000, 2000),
        )
        fig.colorbar(img, aspect=10, pad=0.01, ax=ax[0], format="%+2.f dB")

        # mfcc
        n_mfcc = 12
        extent = [0, time, 0, n_mfcc]
        img = ax[1].imshow(
            np.flipud(mfcc_[:n_mfcc]), aspect="auto", extent=extent, cmap="rainbow"
        )
        ax[1].set(
            title="MFCC sequence", xlabel=None, ylabel="MFCC", yticks=range(0, 13, 4)
        )
        fig.colorbar(img, aspect=10, pad=0.01, ax=ax[1], format="%+2.f dB")

        # delta-MFCC
        d_mfcc = self.delta_mfcc(mfcc_, k=2)

        img = ax[2].imshow(
            np.flipud(d_mfcc[:n_mfcc]), aspect="auto", extent=extent, cmap="rainbow"
        )
        ax[2].set(
            title="ΔMFCC sequence", xlabel=None, ylabel="ΔMFCC", yticks=range(0, 13, 4)
        )
        fig.colorbar(img, aspect=10, pad=0.01, ax=ax[2], format="%+2.f dB")

        ##deltadelta-mfcc
        dd_mfcc = self.delta_mfcc(d_mfcc, k=2)
        img = ax[3].imshow(
            np.flipud(dd_mfcc[:n_mfcc]), aspect="auto", extent=extent, cmap="rainbow"
        )
        ax[3].set(
            title="ΔΔMFCC sequence",
            xlabel="Time [s]",
            ylabel="ΔΔMFCC",
            yticks=range(0, 13, 4),
        )
        fig.colorbar(img, aspect=10, pad=0.01, ax=ax[3], format="%+2.f dB")

        plt.savefig(f"{filename[:-4]}.png")
        plt.show()

