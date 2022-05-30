import numpy as np
import matplotlib.pyplot as plt


def cepstrum(data):
    """
    Measure the cepstrum of the data
    Args:
        data (np.ndarray): Audio data
    Returns:
        cep_res (np.ndarray): Cepstrum of the data
    """
    cep_res = np.fft.ifft(np.log10(np.abs(np.fft.fft(data)))).real
    return cep_res


def f0_cepstrum(data, window=1024, sr=16000, N=512, lifter=32):
    """
    Find the fundamental frequency F0 using the cepstrum method
    Args:
        data (np.ndarray): Wave data
        window (int): Window size, default=1024
        sr (int): Sampling frequency, default=16000
        N (int): Shift length, default=512
        lifter (int): Cutoff lift, default=32
    Returns:
        f0_ceps(np.ndarray): Fundamental frequency F0 using cepstrum
    """
    n_shift = len(data) // N-1
    f0_ceps = np.zeros(n_shift)
    for i in range(n_shift):
        audio = data[i*N:(i*N)+window]*np.hamming(window)
        ceps = cepstrum(audio)
        peak = np.argmax(ceps[lifter:len(ceps)//2])
        f0_ceps[i] = sr/(peak+lifter)
    return f0_ceps


def auto_correlation(data):
    """
    Compute the auto correlation function of the data.
    Args:
        data (np.ndarray): Audio data
    Returns:
        auto_cor (np.ndarray): Auto Correlation function result.
    """
    auto_cor = np.zeros(len(data))
    for n in range(len(data)):
        auto_cor[n] += data[:len(data) - n]@data[n:]
    return auto_cor


def f0_auto_correlation(data, window=1024, sr=16000, N=256):
    """
    Find the fundamental frequency F0 using the autocorrelation function
    Args:
        data (np.ndarray): Audio data
        window (int): Window size, default=1024
        sr (int): Sampling frequency, default=16000
        N (int): Shift length, default=512
    Returns:
        f0_autocorr (np.ndarray): Fundamental frequency F0 using auto-correlation
    """
    n_frame = len(data) // N-1
    f0_autocorr = np.zeros(n_frame)
    for i in range(n_frame):
        tmp = auto_correlation(data[i * N: i * N + window])*np.hamming(window)
        tmp = tmp[len(tmp) // 2:]
        min_index = np.argmin(tmp)
        max_index = np.argmax(tmp[min_index:]) + min_index
        f0_autocorr[i] = (sr / max_index)
    return f0_autocorr


def f0_plot(data, window=1024, N=512, sr=16000, lifter=32):
    """
    Plot Fundamental Frequency with audio spectrogram
    Args:
        data (np.ndarray): Windowed wave data
        window (int): Window size, default=1024
        sr (int): Sampling frequency, default=16000
        N (int): Shift length, default=512
    Returns:

    """
    # Auto-correlation Method
    f0_autocorr = f0_auto_correlation(data, window, sr, N)
    wave_time = data.size/sr
    t1 = np.linspace(0, wave_time, len(f0_autocorr))

    # Cepstrum Method
    f0_ceps = f0_cepstrum(data, window, sr, N, lifter)
    t2 = np.linspace(0, wave_time, len(f0_ceps))

    # Spectrogram
    plt.figure(figsize=(12, 5))
    plt.title("F0 comparison Spectrogram", size=18)
    plt.xlabel("Time [sec]", size=12)
    plt.ylabel("Frequency [Hz]", size=12)
    plt.specgram(data, Fs=sr, cmap='viridis')
    cbar = plt.colorbar()
    cbar.set_label("Magnitude [dB]", size=9)
    plt.plot(t1, f0_autocorr, label='F0 by Auto Correlation', color='blue')
    plt.plot(t2, f0_ceps, label='F0 by Cepstrum', color='black')
    plt.legend()
    plt.xlim(0, wave_time)
    plt.savefig('Fundamental_freq.png')
    plt.show()
    plt.close()
