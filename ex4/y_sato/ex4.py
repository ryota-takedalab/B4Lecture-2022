import librosa
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import soundfile as sf
import matplotlib.ticker
import stft
import scipy


def autocorrelation(signal, window=1024):
    spec = np.fft.rfft(signal, window*2)
    power = np.abs(spec) ** 2
    ac = np.fft.irfft(power)
    ac = ac[: window]
    return ac


def peak(data, threshold=0):
    peak = []
    # exclude the beginning(first peak) and the end
    for i in range(threshold, data.shape[0] - 2):
        if data[i] < data[i + 1] and data[i + 1] > data[i + 2]:
            peak.append([i + 1, data[i + 1]])
    return np.array(peak)


def calc_ac(signal, window=1024):
    Z = []
    for i in range((signal.shape[0] - window) // step):
        #ループはSTFTと同じ雰囲気
        r = autocorrelation(signal[i*step : i*step + window], window=window)
        peaks = peak(r)
        p = peaks[np.argmax(peaks, axis=0)][1][0]
        f0 = sr / p
        Z.append(f0)
    Z = np.array(Z)
    return Z


def cepstrum(signal, threshold, sr, window):
    #フーリエ変換
    win_fc = np.hamming(window)
    tmp = signal * win_fc
    spec = np.fft.fft(tmp)
    log_spec = 20 * np.log10(spec)
    tmp = 10 ** (log_spec / 20)


    #ケプストラム
    cep = np.fft.fft(log_spec)
    #cep = cep.real
    i_cep = np.fft.ifft(spec)

    #f0
    peaks = peak(cep[: window // 2], threshold=threshold)
    m = peaks[np.argmax(peaks, axis=0)][1][0]
    f0 = sr / m

    #env
    env = np.copy(cep)
    env[threshold : window - threshold] = 0
    env = np.fft.ifft(env, axis=0)

    #micro
    micro = np.copy(cep)
    micro[:threshold] = 0
    micro[-threshold:] = 0
    micro = np.fft.ifft(micro, axis=0)

    return f0, env, micro


def lpc(signal, p, sr, window=1024):
    ac = autocorrelation(signal, window)
    r = ac[: p + 1]  # r0 ~ rp
    a, e = levinson_durbin(r)
    w, h = scipy.signal.freqz(e, a)
    w = sr * w / 2 / np.pi
    env = 20 * np.log10(np.abs(h))
    return w, env


def levinson_durbin(r):
    alpha = np.zeros_like(r)
    alpha[0] = 1.0 #これで添え字がインデックスと揃う
    alpha[1] = -r[1] / r[0]
    sigma = r[0] + r[1] * alpha[1]
    for p in range(1, alpha.shape[0]):
        w = np.sum(alpha[: p + 1] * r[p :: -1])
        k = w / sigma #kp = wp /sigmap
        sigma = sigma - k * w #sigma(p+1) = sigma(p) - kp*wp
        alpha[1 : p + 1] = alpha[1 : p + 1] - k * alpha[p - 1  :: -1] #a(p+1) = a(p) - kpa(p)
    e = np.sqrt(sigma)
    return alpha, e


def calc_cep(signal, threshold, sr, window=1024):

    f0series = []
    envseries = []
    microseries = []

    for i in range((signal.shape[0] - window) // step):
        tmp_signal = signal[i*step : i*step + window]
        f0, env, micro = cepstrum(tmp_signal, threshold, sr, window)
        f0series.append(f0)
        envseries.append(env)
        microseries.append(micro)
    f0series = np.array(f0series)
    envseries = np.array(envseries)
    microseries = np.array(microseries)
    return f0series, envseries, microseries





def spectrogram(ax, spec, frame_length, sr, window):
    spec_log = 20 * np.log10(np.abs(spec).T)[window // 2:] #dB
    im = ax.imshow(spec_log, cmap='jet', extent=[0, frame_length // sr, 0, sr // 2,], aspect="auto")
    #ax.set_yscale("log", base=2)
    #ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim([0, 1000])
    ax.set_xlabel('Time[s]')
    ax.set_ylabel('Frequency[Hz]')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', '2%', pad=0.1)
    cbar = fig.colorbar(im, format='%+2.0f dB', cax=cax)
    cbar.set_label("Magnitude[dB]")
    ax.set_title("Spectrogram")


if __name__ == "__main__":
    #load file
    file_name = "audio.wav"
    window = 2048
    step = window // 2

    #original_signal = 音声信号の値、sr=サンプリング周波数 を取得
    original_signal, sr = librosa.load(file_name, sr=None)
    frame_length = original_signal.shape[0]

    #時間軸
    time = np.arange(0, original_signal.shape[0]) / sr

    #STFT
    original_spec = stft.stft(original_signal, window, step)

    #PLOT
    fig = plt.figure(figsize=(8, 6))

    #Original Spectrogram
    ax1 = fig.add_subplot(111)
    spectrogram(ax1, original_spec, frame_length, sr, window)

    ax1.set_title("Fundamental Frequency")
    # f0 (autocorrelation)
    f0_ac = calc_ac(original_signal, window=window)
    t = np.linspace(0, frame_length // sr, f0_ac.shape[0])
    ax1.plot(t, f0_ac, color='black', label="Autocorrelation")

    # f0 (cepstrum)
    threshold = 50
    f0, env, micro = calc_cep(original_signal, threshold, sr, window=window)
    t = np.linspace(0, frame_length // sr, f0.shape[0])
    ax1.plot(t, f0, color='blue', label="Cepstrum")

    ax1.legend()
    plt.show()
    """
    #Cepstrum of whole signal
    env = 10 ** (env / 20)
    micro = 10 ** (micro / 20)

    ispec_env = stft.istft(env, frame_length, window, step)
    ispec_micro = stft.istft(micro, frame_length, window, step)

    #synthesize sound
    sf.write("env.wav", ispec_env, sr, subtype="PCM_16")
    sf.write("micro.wav", ispec_micro, sr, subtype="PCM_16")

    ax3 = fig.add_subplot(3, 2, 3)
    spectrogram(ax3, env, frame_length, sr, window)
    ax4 = fig.add_subplot(3, 2, 4)
    spectrogram(ax4, micro, frame_length, sr, window)
    """
    #再合成：動かない
    #resynthesized = stft.convolution(ispec_env, ispec_micro)
    #sf.write("resynthesized.wav", resynthesized, sr, subtype="PCM_16")

    #Focus on a certain time (t[s])
    fig = plt.figure(figsize=(8, 6))
    t = 2
    t_sample = t * sr
    threshold = 50
    signal = original_signal[t_sample : t_sample + window]
    win_fc = np.hamming(window)

    tmp = signal * win_fc
    t_original_spec = np.fft.fft(tmp)
    t_log_spec = 20 * np.log10(t_original_spec)

    cep_f0, cep_env, micro = cepstrum(signal, threshold, sr, window)
    lpc_w, lpc_env = lpc(signal, 32, sr, window)

    #Plot
    ax2 = fig.add_subplot(111)
    x2 = np.linspace(0, sr // 2, window // 2)
    ax2.plot(x2, t_log_spec[: window // 2 ], label="Original")
    ax2.plot(x2, cep_env[: window // 2 ], label="Cepstrum")
    ax2.plot(lpc_w, lpc_env, label="LPC")
    ax2.set(title="Spectral Envelope", xlabel="Frequency(Hz)", ylabel="Amplitude(dB)")
    ax2.set_xlim([0,10000])
    ax2.legend()


    plt.tight_layout()
    plt.show()
