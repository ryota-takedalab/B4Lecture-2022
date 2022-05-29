import numpy as np
from scipy import signal

def autocorrelation(data, lim):
    r = np.zeros(len(data))
    for m in range (lim):
        r[m] = (data[:lim-m]*data[m:lim]).sum()
    # print(r)  # [1.04638714 1.04051648 1.0285921  ... 0.         0.         0.        ]
    return r

def calc_ac(data, shift_size, samplerate):
    overlap = shift_size//2
    # 窓を適用する回数
    shift = int((data.shape[0] - overlap) / overlap)
    # print('shift', shift)  # 134
    f0 = np.zeros(shift)
    win = np.hamming(shift_size)

    for t in range(shift):
        shift_data = data[t*overlap : t*overlap + shift_size] * win
        # auto correlation
        r = autocorrelation(shift_data, shift_data.shape[0])
        # peak
        m0 = detect_peak(r[: len(r)//2])
        if m0 == 0:
            f0[t] = 0
        else:
            f0[t] = samplerate / m0

    return f0

def cepstrum(data):
    fft_data =np.fft.fft(data)
    power_spec = np.log10(np.abs(fft_data))
    #cep = np.real(np.fft.fft(power_spec))
    cep = np.real(np.fft.ifft(power_spec))  # why ifft?
    return cep


def calc_cep(data, shift_size, samplerate):
    overlap = shift_size//2
    shift = int((data.shape[0] - overlap)/overlap)
    # print('shift', shift)  # 134
    f0 = np.zeros(shift)
    win = np.hamming(shift_size)

    for t in range(shift):
        shift_data = data[int(t*overlap):int(t*overlap+shift_size)] * win
        # auto correlation
        cep = cepstrum(shift_data)
        # peak
        m0 = detect_peak(cep)
        if m0 == 0:
            f0[t] = 0
        else:
            f0[t] = samplerate / m0

    return f0

def detect_peak(r):
    peak=np.zeros(r.shape[0]-2)
    for i in range(r.shape[0]-2):
        if r[i]<r[i+1] and r[i+1]>r[i+2]:
            peak[i] = r[i+1]
    m0 = np.argmax(peak)
    return m0

def levinson_durbin(r, order):

    a = np.zeros(order+1)
    k = np.zeros(order)
    a[0] = 1
    a[1] = -r[1] /r[0]
    k[0] = a[1]
    e = r[0] + r[1] * a[1]
    for q in range(1, order):
        k[q] = -np.sum(a[:q+1] * r[q+1:0:-1]) / e  # kの定義
        U = a[0:q+2]  # aの一番下に0を追加したベクトル
        V = U[::-1]  # Uの上下を逆にした行列
        a[0:q+2] = U + k[q] * V  # A(p+1) = U(p+1) + k(p)*V(p+1)
        e *= 1-k[q] * k[q]  # E(p+1) = E(p)(1-k(p)^2)

    return a, e


def lpc(data, order, shift_size):
    r = autocorrelation(data, data.shape[0])
    a, e = levinson_durbin(r[:len(r) // 2], order)
    # print('a',a)

    h = signal.freqz(np.sqrt(e), a, shift_size, 'whole')[1]  # 指数変換
    env_lpc = 20*np.log10(np.abs(h))
    return env_lpc