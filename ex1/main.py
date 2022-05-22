from wave import Wave_read
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import soundfile as sf
#import warnings

#warnings.filterwarnings("ignore")

def stft(wav, hop, win_length):
    hop_length = int(win_length * hop)
    window = np.hamming(win_length)
    spec = []
    for j in range(0, len(wav), hop_length):
        x = wav[j:j + win_length]
        #print("len(x):", len(x))
        if win_length > len(x):
            break
        x = window * x
        x = np.fft.fft(x)
        #print("fft(x).shape:", x.shape)
        spec.append(x)
    spec = np.array(spec).T
    return spec

#inverse stft
def istft(spec, hop, win_length):

    '''
    istft:transform data by inverse short-time Fourier transform

    Parameters
    ----------
    data: ndarray
        complex-valued spectrogram
    win_length: int
        window length
    hop: float
        hop size

    Returns
    -------
    wave_data: ndarray
        waveform data
    '''
    hop_length = int(win_length * hop)
    ite = spec.shape[0]
    print("spec.shape[0]:", spec.shape[0])
    print("win_length:", win_length)
    window = np.hamming(win_length)

    wave_data = np.zeros(ite * hop_length + win_length)
    for i in range(ite):
        x = spec[i]
        x = np.fft.ifft(x) * win_length
        wave_data[i * hop_length * hop_length + win_length] = wave_data[i * hop_length:i * hop_length + win_length] + x
    
    return wave_data

def main():
    dir = os.path.dirname(__file__) + "/"
    filename = dir + "voice_recording.wav"

    wav, sr = librosa.load(filename, mono = True)

    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
    plt.subplots_adjust(hspace=0.7)

    librosa.display.waveshow(wav, sr = sr, color = "g", ax = ax[0])
    ax[0].set(title = "Original signal", xlabel = None, ylabel = "Amplitude")

    hop = 0.5
    win_length = 1024
    hop_length = int(win_length * hop)

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
    inv_wav = istft(spec, hop = hop, win_length = win_length)
    librosa.display.waveshow(inv_wav, sr = sr, color = "g", ax = ax[2])
    ax[2].set(title = "Re-synthesized signal", xlabel = "Time [s]", ylabel = "Magnitude")

    #graph adjustment
    ax_pos_0 = ax[0].get_position()
    ax_pos_1 = ax[1].get_position()
    ax_pos_2 = ax[2].get_position()
    ax[0].set_position([ax_pos_0.x0, ax_pos_0,y0, ax_pos_1.width, ax_pos_1.height])
    ax[2].set_position([ax_pos_2.x0, ax_pos_2.y0, ax_pos_1.width, ax_pos_1.height])
    fig.align_labels()

    #save and show figure of result
    plt.savefig(dir + "ex1_result.png")
    plt.show()

if __name__ == "__main__":
    main()
