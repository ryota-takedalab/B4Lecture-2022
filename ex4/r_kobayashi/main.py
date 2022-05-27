import argparse
import os

import librosa
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ssig

import warnings

warnings.filterwarnings("ignore")

# 自己相関関数
def autocorrelation(data, order=None):
    """
    parameter
    ---
    data:numpy.ndarray
         audio time series
    
    returns
    ---
    ac:numpy.ndarray
       autocorrelation
    """
    len_data = len(data)
    ac = np.zeros(len_data)
    if order == None:
        order = len_data
    for l in range(order):
        for i in range(len_data - 1):
            ac[i] += data[i] * data[i+1]
    
    return ac


# 自己相関関数のピーク
def peak(ac):
    """ 
    parameter
    ---
    ac:numpy.ndarray
       autocorrelation
       
    return
    ---
    m:int
      peak of input
    """
    peak = np.zeros(ac.shape[0] - 2)
    # 前後で比較
    for i in range(ac.shape[0] - 2):
        if ac[i]<ac[i+1] and ac[i+1]>ac[i+2]:
            peak[i] = ac[i+1]
    m0 = np.argmax(peak)

    return m0


def f0_ac(data, sr, F_size, overlap, S_num):
    """ 
    parameters
    ---
    data:numpy.ndarray
         audio time series
    sr:int
       sampling rate
    F_size:int
           frame size
    overlap:int
            overlap size
    S_num:int
          a number of shift
    
    return
    ---
    f0_ac:np.ndarray
          fundamental frequency
    """
    win = np.hamming(F_size)
    f0_ac = np.zeros(S_num)
    for i in range(S_num):
        windata = data[i*overlap : (i*overlap + F_size)] * win
        cor = autocorrelation(windata)
        peak_m = peak(cor[:len(cor)//2])
        if peak_m == 0:
            f0_ac[i] = 0
        else:
            f0_ac[i] = sr / peak_m

    return f0_ac


def cepstrum(data):
    """ 
    parameter
    ---
    data:numpy.ndarray
            audio time series
    
    return
    ---
    cep:numpy.ndarray
        cepstrum
    """
    fft_data = np.fft.fft(data)
    power_spec = np.log10(np.abs(fft_data))
    cep = np.real(np.fft.ifft(power_spec))

    return cep


def f0_cep(data, sr, F_size, overlap, S_num, lif):
    """ 
    parameters
    ---
    data:numpy.ndarray
         audio time series
    sr:int
       sampling rate
    F_size:int
           frame_size
    overlap:int
            overlap size
    S_num:int
          a number of frame
    lif:int
        lifter index
    
    return
    ---
    f0:np.ndarray
       fundamental frequency
    """
    win = np.hamming(F_size)
    f0 = np.zeros(S_num)
    # len_data = len(data)
    for i in range(S_num):
        data_ = data[i*overlap : (i*overlap + F_size)] * win
        cep = cepstrum(data_)
        m0 = np.argmax(cep[lif:len(cep)//2]) + lif
        f0[i] = sr / m0
    
    return f0

# 基本周波数のプロット
def f_plot(f0_a, f0_c, data, Ts, sr):
    """ 
    parameters
    ---
    f0_a and f0_c:np.ndarray
                  fundamental frequency
    data:numpy.ndarray
         audio time series
    sr:int
       sampling rate
    Ts:float
       time of sound data
    sr:int
       sampling rate
    """
    plt.figure(figsize=(10, 8))
    plt.specgram(data, Fs=sr, cmap="rainbow", scale_by_freq="True")
    xa_axis = np.arange(0, Ts, Ts/len(f0_a))
    xc_axis = np.arange(0, Ts, Ts/len(f0_c))
    plt.plot(xa_axis, f0_a, label="Autocorrelation", color="blue")
    plt.plot(xc_axis, f0_c, label="Cepstrum", color="green")
    plt.xlabel("times[s]", fontsize=15)
    plt.ylabel("Frequency[Hz]", fontsize=15)
    plt.legend()
    plt.colorbar()
    plt.tight_layout()
    path = os.path.dirname(__file__)
    save_f0_fname = os.path.join(path, "result", args.save_f0_fname)
    plt.savefig(save_f0_fname)
    plt.show()
    plt.close()


def cep_m(data, lif):
    """
    parameters
    --- 
    data:numpy.ndarray
         audio time series
    lif:int
        lifter index
    
    return
    ---
    cep_env:numpy.ndarray
            spectral envelop
    """
    cep = cepstrum(data)
    cep[lif : len(cep)-lif] = 0
    cep_env = 20 * np.real(np.fft.fft(cep))

    return cep_env

def LevinsonDurbin(r, deg):
    """
    parameters
    ---
    r:numpy.ndarray
      autocorrelation
    deg:int
          degree
    
    returns
    ---
    a:numpy.ndarray
      lpc coefficient
    e:numpy.ndarray
      minimum error
    """
    a = np.zeros(deg + 1)
    e = np.zeros(deg + 1)

    a[0] = 1.0
    a[1] = - r[1] / r[0]
    e[1] = r[0] + r[1]*a[1]
    lam = - r[1] / r[0]

    for k in range(1, deg):
        lam = 0.0
        for j in range(k + 1):
            lam -= a[j] * r[k + 1 - j]
            lam /= e[k]

        U = [1]
        U.extend([a[i] for i in range(1, k + 1)])
        U.append(0)

        V = [0]
        V.extend([a[i] for i in range(k, 0, -1)])
        V.append(1)

        a = np.array(U) + lam * np.array(V)

        e[k + 1] = e[k] * (1.0 - lam**2)

    return a, e[-1]


def preemphasis(data, p):
    """ 
    parameters
    ---
    data:numpy.ndarray
         audio time series
    p:int
      filter coefficient
      
    return
    ---
    f:numpy.ndarray
      FIR filter
    """
    f = ssig.lfilter([1.0, -p], 1, data)

    return f


def lpc_m(data, deg, F_size):
    """ 
    parameters
    ---
    data:numpy.ndarray
         audio time series
    deg:int
        degree
    F_size:int
           frame size
           
    return
    ---
    lpc_env:numpy.ndarray
            spectral envelop
    """
    r = autocorrelation(data, deg + 1)
    r = r[:len(r)//2]
    a, e = LevinsonDurbin(r, deg)

    w, h = ssig.freqz(np.sqrt(e), a, F_size, "whole")
    lpc_env = 20 * np.log10(np.abs(h))

    return lpc_env

# スペクトル包絡のプロット
def spe(log, cep, lpc, F_size, sr):
    """
    parameters
    ---
    log:numpy.ndarray
        spectrum
    cep:numpy.ndarray
        spectral envelop
    lpc:numpy.ndarray
        spectral envelop
    F_size:int
           frame size
    sr:int
       sampling rate
    """
    plt.figure(figsize=(10, 8))
    f_axis = np.fft.fftfreq(F_size, d=1.0/sr)
    plt.plot(f_axis[:F_size//2], log[:len(log)//2], label="Spectrum", color="blue")
    plt.plot(f_axis[:F_size//2], cep[:len(log)//2], label="Cepstrum", color="green")
    plt.plot(f_axis[:F_size//2], lpc[:len(log)//2], label="LPC", color="red")
    plt.xlabel("Frequency[Hz]", fontsize=15)
    plt.ylabel("Log amplitude spectrum[dB]", fontsize=15)
    plt.legend()
    plt.tight_layout()
    path = os.path.dirname(__file__)
    save_spe_fname = os.path.join(path, "result", args.save_spe_fname)
    plt.savefig(save_spe_fname)
    plt.show()
    plt.close()


def main(args):
    path = os.path.dirname(__file__)
    fname = os.path.join(path, args.fname)
    data, sr = librosa.load(fname, sr = 16000)

    # 基本周波数計算
    F_size = 1024
    overlap = F_size // 2
    F_num = data.shape[0]
    Ts = float(F_num)
    S_num = int(F_num//(F_size - overlap) - 1)
    win = np.hamming(F_size)
    lif = args.lif

    # 基本周波数計算&プロット
    f0_a = f0_cep(data, sr, F_size, overlap, S_num, lif)
    f0_c = f0_ac(data, sr, F_size, overlap, S_num)
    f_plot(f0_a, f0_c, data, Ts, sr)

    # スペクトル包絡
    p = 0.97
    s = 2.0
    s_frame = int(s * sr)
    pe_data = preemphasis(data, p)
    windata = pe_data[s_frame:s_frame + F_size] * win
    deg = args.deg

    # 計算
    log = 20 * np.log10(np.abs(np.fft.fft(windata)))
    cep = cep_m(windata, lif)
    lpc = lpc_m(windata, deg, F_size)

    spe(log, cep, lpc, F_size, sr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fname", type = str, help = "Load file name")
    parser.add_argument("--lif", default=32)
    parser.add_argument("--deg", default=64)
    parser.add_argument("-s1", "--save_f0_fname", type = str, help = "Save f0 file name")
    parser.add_argument("-s2", "--save_spe_fname", type = str, help = "save spectral envelop file name")
    args = parser.parse_args()

    main(args)