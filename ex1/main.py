from wave import Wave_read
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import soundfile as sf
import warnings

warnings.filterwarnings("ignore")

def stft(wav, hop, win_length):
    hop_length = int(win_length * hop)
    window = np.hamming(win_length)
    
    # padding zero
    padding_length = win_length - len(wav) % (hop_length)
    wav = np.concatenate([wav, np.zeros(padding_length)])
    
    # スペクトログラムの配列生成
    spec = np.empty((len(wav) // hop_length -1, win_length))
    
    # stft
    for i in range(0, spec.shape[0] - 1):
        x = wav[i * hop_length : i * hop_length + win_length]
        x = window * x 
        spec[i] = np.fft.fft(x)
        #print("spec.shape:", spec.shape)
    spec = np.array(spec).T    
    return spec

#inverse stft
def istft(spec, hop, win_length):

    '''
    istft:transform data by inverse short-time Fourier transform

    Parameters
    ----------
    spec: ndarray
        complex-valued spectrogram
    win_length: int
        window length
    hop: float
        hop size

    Returns
    -------
    inv_wav: ndarray
        waveform data
    '''
    hop_length = int(win_length * hop)
    inv_wav = np.empty(hop_length * (spec.shape[0] + 1))

    # istft
    for i in range(0, spec.shape[0] - 1):
        inv_wav[
            i * hop_length : i * hop_length + win_length
        ] = np.fft.ifft(spec[i]).real

    return inv_wav

def main():
    dir = os.path.dirname(__file__) + "/"
    filename = dir + "original_wave.wav"

    wav, sr = librosa.load(filename, mono = True)

    # グラフ設定
    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
    plt.subplots_adjust(hspace=0.7)

    librosa.display.waveshow(wav, sr = sr, color = "g", ax = ax[0])
    ax[0].set(title = "Original signal", xlabel = None, ylabel = "Amplitude")

    hop = 0.5
    win_length = 64
    hop_length = int(win_length * hop)

    # stft
    spec = stft(wav, hop = hop, win_length = win_length)
    db = librosa.amplitude_to_db(np.abs(spec))
    img = librosa.display.specshow(
        db,
        sr = sr,
        hop_length = hop_length,
        x_axis = "time",
        y_axis = "log",
        ax = ax[1],
        cmap = "plasma",
    )
    ax[1].set(title = "Spectrogram", xlabel = None, ylabel = "Frequency [Hz]")
    ax[1].set_yticks([0, 128, 512, 2048, 8192])
    fig.colorbar(img, aspect = 10, pad = 0.05, extend = "both", ax = ax[1], format = "%+2.f dB")

    #inverse-stft
    inv_wav = istft(np.array(spec).T, hop = hop, win_length = win_length)
    librosa.display.waveshow(inv_wav, sr = sr, color = "g", ax = ax[2])
    ax[2].set(title = "Re-synthesized signal", xlabel = "Time [s]", ylabel = "Amplitude")
    sf.write(dir + "\\result\\inverse_wave.wav", inv_wav, sr)

    #graph adjustment
    ax_pos_0 = ax[0].get_position()
    ax_pos_1 = ax[1].get_position()
    ax_pos_2 = ax[2].get_position()
    ax[0].set_position([ax_pos_0.x0, ax_pos_0.y0, ax_pos_1.width, ax_pos_1.height])
    ax[2].set_position([ax_pos_2.x0, ax_pos_2.y0, ax_pos_1.width, ax_pos_1.height])
    fig.align_labels()

    #save and show figure of result
    plt.savefig(dir + "\\result\\ex1_result.png")
    plt.show()

if __name__ == "__main__":
    main()
