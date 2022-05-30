import numpy as np
import scipy.signal
import scipy.linalg
import matplotlib.pyplot as plt

from .import cepstrum
from . import f0


def log_spectrum(data):
    """log spectrum

    Args:
        data (ndarray, axis=(time, )): input data

    Returns:
        ndarray, axis=(frequency, ): log amplitude
    """
    spectrum = np.fft.fft(data * np.hanning(len(data)))
    return 20 * np.log10(np.abs(spectrum))


def envelope_cepstrum(data):
    """spectrum envelope based on cepstrum

    Args:
        data (ndarray, axis=(time, )): input data

    Returns:
        ndarray, axis=(frequency, ): spectram envelope based on cepstrum
    """
    data_cepstrum = cepstrum.cepstrum(data)
    lp_lifter = cepstrum.craete_lifter(len(data), cutoff_frame=20)
    return np.real(np.fft.fft(data_cepstrum * lp_lifter))


# TODO: おかしい
def envelope_lpc(data, p, fs):
    data = data * np.hanning(len(data))
    # TODO: docstring
    # auto_correlation = f0.auto_correlation(data)
    auto_correlation = np.correlate(data, data, mode="full")
    auto_correlation = auto_correlation[auto_correlation.size // 2:]
    # alphas, e = LevinsonDurbin(auto_correlation, p)
    # alphas, e = levinson_durbin(auto_correlation, p)
    # alphas = solve_toeplitz(auto_correlation[:p],
    #                         -auto_correlation[1:p + 1])
    # toe = scipy.linalg.toeplitz(auto_correlation[:p])
    # alphas = -np.linalg.inv(toe) @ auto_correlation[1:p + 1]
    alphas = scipy.linalg.solve_toeplitz(auto_correlation[:p],
                                         -auto_correlation[1:p + 1])
    alphas = np.append([1], alphas)  # add constant term
    print(alphas)
    w, h = scipy.signal.freqz(1, alphas, whole=True, fs=fs)
    
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111)
    # ax.plot(log_spectrum(h))
    # fig.show()
    return 20 * np.log10(np.abs(h))


# TODO: おかしい
def solve_toeplitz(x, b):
    dimension = len(b)
    if (dimension != len(x)):
        raise ValueError("wrong shape")
    answers = np.zeros(dimension + 1)
    residual_error = np.zeros(dimension + 1)
    
    if False:
        toeplitz = np.zeros((dimension, dimension))
        for i in range(dimension):
            toeplitz[i:, i] = x[:dimension - i]
            toeplitz[i, i:] = x[:dimension - i]
        print(toeplitz)
        # とりあえず愚直に逆行列で…
        answers = np.linalg.inv(toeplitz) @ -b
    elif True:
        # base stage
        answers[0] = 1
        answers[1] = -x[1] / x[0]
        residual_error[0] = x[0] + answers[1] * x[1]
        # print(f"{toeplitz[0, 0]} + {answers[0]} * {toeplitz[1, 0]}")
        
        # recursive stage
        for i in range(dimension - 1):
            print(i)
            lambda_ = 0
            for j in range(i + 1):
                lambda_ -= answers[j] * b[i + 1 - j]  # NOTE: 怪しい
            lambda_ /= residual_error[i]
            print(np.concatenate([answers[: i + 1], [0]]),

                                np.concatenate([[0], np.flip(answers[1: i + 1]), [1]]))
            
            answers[: i + 2] = (np.concatenate([answers[: i + 1], [0]]) +
                                lambda_ *
                                np.concatenate([[0], np.flip(answers[1: i + 1]), [1]]))
            residual_error[i + 1] = (1 - np.power(lambda_, 2)) * residual_error[i]
    else:
        # k = 1の場合
        answers[0] = 1.0
        answers[1] = - x[1] / x[0]
        residual_error[1] = x[0] + x[1] * answers[1]
        lam = - x[1] / x[0]

        # kの場合からk+1の場合を再帰的に求める
        for k in range(1, dimension):
            # lambdaを更新
            lam = 0.0
            for j in range(k + 1):
                lam -= answers[j] * x[k + 1 - j]
            lam /= residual_error[k]

            # aを更新
            # UとVからaを更新
            U = [1]
            U.extend([answers[i] for i in range(1, k + 1)])
            U.append(0)

            V = [0]
            V.extend([answers[i] for i in range(k, 0, -1)])
            V.append(1)

            answers = np.array(U) + lam * np.array(V)

            # eを更新
            residual_error[k + 1] = residual_error[k] * (1.0 - lam * lam)
    return answers[1:]
def autocorrelation_t(data):
    """define auto correlation

     Args:
         data (ndarray): input signal

     Returns:
         r (ndarray): auto correlation signal
     """
    
    r = np.zeros(len(data))
    for m in range (data.shape[0]):
        r[m] = (data[:data.shape[0]-m]*data[m:data.shape[0]]).sum()
    return r

def lpc(data, order, shift_size):
    r = autocorrelation_t(data)
    a, e = levinson_durbin(r[:len(r) // 2], order)
    # print('a',a)

    h = scipy.signal.freqz(np.sqrt(e), a, shift_size, 'whole')[1]  # 指数変換
    env_lpc = 20*np.log10(np.abs(h))
    return env_lpc

def LevinsonDurbin(r, lpcOrder):
    """Levinson-Durbinのアルゴリズム
    k次のLPC係数からk+1次のLPC係数を再帰的に計算して
    LPC係数を求める"""
    # LPC係数（再帰的に更新される）
    # a[0]は1で固定のためlpcOrder個の係数を得るためには+1が必要
    a = np.zeros(lpcOrder + 1)
    e = np.zeros(lpcOrder + 1)

    # k = 1の場合
    a[0] = 1.0
    a[1] = - r[1] / r[0]
    e[1] = r[0] + r[1] * a[1]
    lam = - r[1] / r[0]

    # kの場合からk+1の場合を再帰的に求める
    for k in range(1, lpcOrder):
        # lambdaを更新
        lam = 0.0
        for j in range(k + 1):
            lam -= a[j] * r[k + 1 - j]
        lam /= e[k]

        # aを更新
        # UとVからaを更新
        U = [1]
        U.extend([a[i] for i in range(1, k + 1)])
        U.append(0)

        V = [0]
        V.extend([a[i] for i in range(k, 0, -1)])
        V.append(1)

        a = np.array(U) + lam * np.array(V)

        # eを更新
        e[k + 1] = e[k] * (1.0 - lam * lam)

    return a, e[-1]

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
