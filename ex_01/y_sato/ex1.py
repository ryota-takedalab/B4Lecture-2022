import librosa
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def STFT(signal,window,step):
    Z = []
    win_fc=np.hamming(window)
    for i in range((signal.shape[0] - window) // step):
        tmp = signal[i*step : i*step + window]
        tmp = tmp * win_fc
        tmp = np.fft.fft(tmp)
        Z.append(tmp)
    Z= np.array(Z)
    return Z


def ISTFT(y,frame_length,window,step):
    Z = np.zeros(frame_length)
    for i in range(len(y)) :
        tmp = np.fft.ifft(y[i]) #逆フーリエ変換
        Z[i*step : i*step + window] += np.real(tmp)
    Z= np.array(Z)
    return Z


if __name__ == "__main__":
    #load file
    file_name = "audio.wav"
    window = 1024
    step = window//2

    #original_signal = 音声信号の値、sr=サンプリング周波数 を取得
    original_signal, sr = librosa.load(file_name)
    frame_length = original_signal.shape[0]
    #時間軸
    time = np.arange(0,original_signal.shape[0]) / sr

    #STFT
    spec = STFT(original_signal,window,step)
    #ISTFT
    ispec = ISTFT(spec,frame_length,window,step)

    #PLOT
    fig = plt.figure(figsize=(8,6))

    #Original Signal
    ax1 = fig.add_subplot(3,1,1)
    ax1.plot(time,original_signal)
    ax1.set_xlabel("Time(s)")
    ax1.set_ylabel("Sound Amplitude")
    ax1.set_title("Original Signal")

    #Spectrogram
    ax2 = fig.add_subplot(3,1,2)

    spec_log = 20 * np.log10(np.abs(spec).T)[window // 2:] #dB
    im = ax2.imshow(spec_log,extent = [0,original_signal.shape[0] // sr,0,sr // 2,] ,aspect="auto")

    ax2.set_yscale("log",base=2)
    ax2.set_ylim([50,sr//2])
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Frequency [Hz]')

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', '2%', pad=0.1)
    cbar = fig.colorbar(im, format='%+2.0f dB', cax=cax)
    cbar.set_label("Magnitude [dB]")
    ax2.set_title("Spectrogram")

    ax3 = fig.add_subplot(3,1,3)
    ax3.plot(time,ispec)
    ax3.set_xlabel("Time(s)")
    ax3.set_ylabel("Sound Amplitude")
    ax3.set_title("Re-Synthesized Signal")

    plt.tight_layout()
    plt.show()
