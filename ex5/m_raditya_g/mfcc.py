import numpy as np
from scipy.signal import stft


class Mfcc:
    def __init__(self, data, samplerate, win_length, k, n, f0):
        """
        Parameters used in the class
        :param data: (np.ndarray) Audio data
        :param samplerate: (int) Audio Sample Rate
        :param f0: (int) basis frequency
        :param k: (int) width of delta
        :param n: (int) Filter bank number
        """
        self.sr = samplerate
        self.win_length = win_length
        self.filter_bank = None
        self.mel_spec = None
        self.data = data
        self.f0 = f0
        self.k = k
        self.n = n
        self.m0 = 1000.0 / np.log(1000.0 / self.f0 + 1.0)

    def freq_to_mel(self, f):
        """
        Convert Frequency [Hz] to Mel-frequency
        :return: mel (np.array): converted frequency
        """
        mel = self.m0 * np.log(f / self.f0 + 1.0)
        return mel

    def mel_to_freq(self, mel):
        """
        Convert Frequency [Hz] to Mel-frequency
        :return: f (np.array) = converted mel-frequency
        """
        f = self.f0 * (np.exp(mel / self.m0) - 1.0)
        return f

    def melfilterbank(self):
        """
        Make a mel-filter bank
        :return: filter_bank (np.ndarray): mel filter bank
        """
        # Nyquist frequency
        fn = self.sr / 2.0
        melmax = self.freq_to_mel(fn)
        mel_x = np.linspace(0, melmax, self.n + 2)
        flt_size = (self.win_length >> 1) + 1  # filter size

        # place points at equal intervals of mel scale
        freq_x = np.round(self.mel_to_freq(mel_x) * flt_size / fn).astype(int)

        filter_bank = np.zeros((self.n, flt_size))  # initialize
        for i in range(self.n):
            # make triangle wave
            filter_bank[i, freq_x[i]:freq_x[i + 1]] = np.linspace(0, 1, freq_x[i + 1] - freq_x[i])
            filter_bank[i, freq_x[i + 1]:freq_x[i + 2]] = np.linspace(1, 0, freq_x[i + 2] - freq_x[i + 1])
        self.filter_bank = filter_bank
        return self.filter_bank

    def dct_x(self):
        """
        Calculate the Discrete Cosine Transformation
        :return: dct_res (np.ndarray): Transformed Data
        """
        shape = np.array(self.mel_spec.shape)
        shape[0] = shape[0]*4  # multiply by 4

        # if x = [1, 2, 3]
        # x_4n = [0, 1, 0, 2, 0, 3, 0]
        y = np.zeros(shape)
        print(self.mel_spec.shape[0])
        y[1:self.mel_spec.shape[0]*2:2] = self.mel_spec

        # calculate the DCT[x] (= Re[FFT[y]])
        dct_x = np.fft.fft(y, axis=0)[:self.mel_spec.shape[0]].real

        # normalize
        dct_x[0] *= np.sqrt(1 / self.mel_spec.shape[0])
        dct_x[1:] *= np.sqrt(2 / self.mel_spec.shape[0])
        return dct_x

    def mfcc(self, dim):
        """
        Calculate mfcc using dct and filterbank
        :param dim: (int) dimension of mfcc
        :return: mfcc: (np.ndarray) mfcc result
        """
        t, y, spec = stft(self.data, nperseg=256)
        # Apply mel filter bank
        self.mel_spec = 20 * np.log10(self.filter_bank@np.abs(spec))
        # Discrete cosine transform
        dct1 = self.dct_x()
        mfcc = dct1.T[:, :dim]
        return mfcc

    def del_mfcc(self, data):
        """
        :param data: (np.ndarray) mfcc data
        :return: dmfcc: (np.ndarray) delt of mfcc
        """
        l1 = np.arange(-self.k, self.k+1)
        k_sqr = np.sum(l1**2)
        dmfcc = np.zeros_like(data)
        mfcc_pad = np.pad(data, ((self.k, self.k + 1), (0, 0)), 'edge')
        for n in range(len(data)):
            dmfcc[n] = l1@mfcc_pad[n:n+self.k*2+1]
        dmfcc = dmfcc/k_sqr
        return dmfcc
