import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
from scipy.fftpack import dct
from mpl_toolkits.axes_grid1 import make_axes_locatable
import librosa
import stft


class MFCC:
    def __init__(self, sr, f0=700):
        self.sr = sr
        self.f0 = f0
        self.nceps = 12
        self.calc_m0()


    def calc_m0(self):
        self.m0 = 1000.0 / np.log(1000.0 / self.f0 + 1.0)


    def freq2mel(self, f):
        return self.m0 * np.log(f / self.f0 + 1.0)


    def mel2freq(self, m):
        return self.f0 * (np.exp(m / self.m0) - 1.0)


    def mel_filter_bank(self, N, numChannnels):
        sr = self.sr

        fmax = sr / 2
        melmax = self.freq2mel(fmax)
        nmax = N // 2
        df = sr / N

        dmel = melmax / (numChannnels + 1)
        melcenters = np.arange(1, numChannnels + 1) * dmel
        fcenters = self.mel2freq(melcenters)
        indexcenter = np.round(fcenters / df)
        indexstart = np.hstack(([0], indexcenter[0 : numChannnels-1]))
        indexstop = np.hstack((indexcenter[1:numChannnels], [nmax]))
        filterbank = np.zeros((numChannnels, nmax))

        for c in range(numChannnels):
            increment = 1.0 / (indexcenter[c] - indexstart[c])
            for i in range(int(indexstart[c]), int(indexcenter[c])):
                filterbank[c, i] = (i - indexstart[c]) * increment
            decrement = 1.0 / (indexstop[c] - indexcenter[c])
            for i in range(int(indexcenter[c]), int(indexstop[c])):
                filterbank[c, i] = 1.0 - ((i - indexcenter[c]) * decrement)

        self.filterbank = filterbank
        self.fcenters = fcenters
        return filterbank, fcenters


    def mspec(self, signal):
        win_fc = np.hamming(window)
        signal = signal * win_fc
        spec = np.abs(np.fft.fft(signal))
        spec = spec[: spec.shape[0] // 2]
        mspec = spec @ filterbank.T
        return mspec


    def mfcc(self, signal):
        mspec = self.mspec(signal)
        mfcc = dct(20 * np.log10(mspec))[:self.nceps]
        return mfcc


    def calc_mfcc(self, signal, window=2048):
        Z = []
        for i in range((signal.shape[0] - window) // step):
            mfcc = self.mfcc(signal[i*step : i*step + window])
            Z.append(mfcc)
        Z = np.array(Z)
        return Z


    def delta_mfcc(self, mfcc, l=5):
        Z = np.zeros([mfcc.shape[0], self.nceps])
        k_list = np.arange(-l, l + 1)
        k_sq = np.sum(k_list ** 2) #sigma(k=-l~k)k^2
        for m in range(l, mfcc.shape[0] - l):
            for j in range(self.nceps):
                Z[m][j] = np.sum(k_list * mfcc[m - l : m + l + 1, j]) / k_sq

        return Z


def spectrogram(ax, spec, frame_length, sr, window):
    """show spectrogram
    Args:
        ax: axis
        spec: input spectrogram
        frame_length: Length of signal
        sr: sampling rate
        window: Length of window
    """
    spec_log = spec
    #spec_log = 20 * np.log10(np.abs(spec).T)[window // 2:] #dB
    im = ax.imshow(spec_log, cmap='jet', extent=[0, frame_length // sr, 0, sr // 2,], aspect="auto")
    #ax.set_ylim([0, 1000])
    ax.set_xlabel('Time[s]')
    ax.set_ylabel('Frequency[Hz]')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', '2%', pad=0.1)
    cbar = fig.colorbar(im, format='%+2.0f dB', cax=cax)
    cbar.set_label("Magnitude[dB]")
    ax.set_title("Spectrogram")
    return


if __name__ == "__main__":
    sr = 44100

    #load file
    file_name = "audio.wav"
    window = 2048
    step = window // 2

    #original_signal = 音声信号の値、sr=サンプリング周波数 を取得
    original_signal, sr = librosa.load(file_name, sr=None)
    frame_length = original_signal.shape[0]

    #time scale
    time = np.arange(0, original_signal.shape[0]) / sr

    #STFT
    original_spec = stft.stft(original_signal, window, step)

    numChannels = 20
    df = sr / window
    mfcc = MFCC(sr)
    filterbank, fcenters = mfcc.mel_filter_bank(window, numChannels)

    #plot filter bank
    for c in np.arange(numChannels):
        plt.plot(np.arange(0, window / 2) * df, filterbank[c])

    log_spec = 20 * np.log10(np.abs(original_spec).T)[window // 2:]
    mfcc_list = mfcc.calc_mfcc(original_signal)
    d_mfcc = mfcc.delta_mfcc(mfcc_list)
    dd_mfcc = mfcc.delta_mfcc(d_mfcc)

    plt.plot()
    fig = plt.figure()

    #spectrogram
    ax1 = fig.add_subplot(411)
    im = ax1.imshow(log_spec, cmap='jet', extent=[0, frame_length // sr, 0, sr // 2,], aspect="auto")
    ax1.set(xlabel="Time[s]", ylabel="Frequency[Hz]", title="Spectrogram")
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', '2%', pad=0.1)
    cbar = fig.colorbar(im, format='%+2.0f', cax=cax)


    #mfcc
    ax2 = fig.add_subplot(412)
    im = ax2.imshow(mfcc_list.T, cmap='jet', extent=[0, frame_length // sr, 0, mfcc_list.shape[1]], aspect="auto",  origin="lower")
    ax2.set(xlabel="Time[s]", ylabel="MFCC", title="MFCC sequence")
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', '2%', pad=0.1)
    cbar = fig.colorbar(im, format='%+2.0f', cax=cax)


    #delta_mfcc
    ax3 = fig.add_subplot(413)
    im = ax3.imshow(d_mfcc.T, cmap='jet', extent=[0, frame_length // sr, 0, d_mfcc.shape[1]], aspect="auto",  origin="lower")
    ax3.set(xlabel="Time[s]", ylabel="delta-MFCC", title="delta-MFCC sequence")
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', '2%', pad=0.1)
    cbar = fig.colorbar(im, format='%+2.0f', cax=cax)

    #delta-2_mfcc
    ax4 = fig.add_subplot(414)
    im = ax4.imshow(dd_mfcc.T, cmap='jet', extent=[0, frame_length // sr, 0, dd_mfcc.shape[1]], aspect="auto",  origin="lower")
    ax4.set(xlabel="Time[s]", ylabel="dd-MFCC", title="dd-MFCC sequence")
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes('right', '2%', pad=0.1)
    cbar = fig.colorbar(im, format='%+2.0f', cax=cax)



    plt.tight_layout()
    plt.show()
