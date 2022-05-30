import numpy as np
import scipy.signal as signal
from f0_fundamental_freq import *
import time


def levinson_durbin(acr, lpc_deg):
    """
    Levinson Durbin Recursion
    Args:
        acr (numpy.ndarray): The value of autocorrelation function
        lpc_deg (int): LPC function degree
    Returns:
        a (numpy.ndarray): Prediction coefficient
        e (float): Prediction error
    """
    a = np.zeros(lpc_deg+1)
    e = np.zeros(lpc_deg+1)
    # When i=0 and i=1
    a[0] = 1
    a[1] = -acr[1]/acr[0]
    e[0] = acr[0]
    e[1] = acr[0] + acr[1]*a[1]
    # calculate for i>1 to i=lpc_deg
    for i in range(2, lpc_deg+1):
        k = -np.sum(a[:i]*acr[i:0:-1])/e[i-1]
        a[:i+1] += k*a[:i+1][::-1]
        e[i] = e[i-1] * (1-k**2)
    return a, e[-1]


def lpc(data, window, lpc_deg):
    """
    Find the LPC data using Levinson Durbin
    Args:
        data (np.ndarray): Windowed Audio data
        window (int): Width of window
        lpc_deg (int): Degrees of lpc method
    Return
    ------
        lpc_data (np.ndarray): Data of lpc method
    """
    acr = auto_correlation(data)
    acr = acr[:len(acr)//2]
    start = time.time()
    a, e = levinson_durbin(acr, lpc_deg)
    w, h = signal.freqz(np.sqrt(e), a, window, True)
    lpc_data = 20*np.log10(np.abs(h))
    return lpc_data

def cepstrum_envelope(data, lifter=32):
    """
    Calculate the spectrum envelope data
    Args:
        data (np.ndarray): Windowed wave data
        lifter (int): Cutoff lift, default=32
    Returns:
        cep_env_res (numpy.ndarray): Cepstrum envelope data
    """
    cep = cepstrum(data)
    cep[lifter:len(cep)-lifter] = 0
    cep_env_res = 20*np.fft.fft(cep).real
    return cep_env_res


def spectrum_envelope(data, window=1024, sr=16000, lpc_deg=3):
    """
    Plot spectrum envelope
    Args:
        data (np.ndarray): Windowed wave data
        window (int): Width of window
        sr (int): Sample rate
        lpc_deg (int): LPC degree
    Returns:

    """
    # Spectrum, cepstrum, and LPC
    log_data = 20*np.log10(np.abs(np.fft.fft(data)))
    cep_data = cepstrum_envelope(data)
    lpc_data = lpc(data, window, lpc_deg)
    freq = np.fft.fftfreq(window, d=1.0/sr)

    plt.figure(figsize=(10, 10))
    plt.title('Envelope')
    plt.plot(freq[:window//2], log_data[:len(log_data)//2], label='Spectrum', color='blue')
    plt.plot(freq[:window//2], cep_data[:len(log_data)//2], label='Cepstrum', color='yellow')
    plt.plot(freq[:window//2], lpc_data[:len(log_data)//2], label='LPC', color='red')
    plt.xlabel('Frequency[Hz]')
    plt.ylabel('Amplitude[dB]')
    plt.legend()
    plt.savefig('spectrum.png')
    plt.show()
    plt.close()