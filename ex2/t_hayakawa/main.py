import numpy as np
from scipy import signal
import librosa
import librosa.display
from matplotlib import pyplot as plt
import argparse
import soundfile


def STFT(data, window=1024, step=512):
    """
    Short Time Fourier Transform

    Parameters
    ----------
    data : Input Signal (np.aray)
    window : Length of window size
        width of cut signal
    step : Length of step size
    """
    # window function
    win_fc = np.hamming(window)

    frame = (len(data)-window+step)//step

    framed_signal = np.lib.stride_tricks.as_strided(data, shape=(
        window, frame), strides=(data.strides[0], data.strides[0]*step))

    # win_fc.shape=(window,) -> win_fc[:,np.newaxis].shape=(window,1)
    window_signal = framed_signal*win_fc[:, np.newaxis]
    # fast fourier transform
    spec = np.fft.rfft(window_signal, axis=0)
    # -----------

    return spec

def Convolution(input, filter):
    """
    Parameters
    ----------
    input: ndarray
    filter: ndarray

    Return
    ----------
    result: ndarray
    """

    result = np.zeros(input.size+filter.size-1)

    for i in range(input.size):
        result[i:i+filter.size] += input[i]*filter

    # -----------------------
    return result


def HPF(freq, sr, window):
    """
    High Pass Filter

    Parameters
    ----------
    freq: int
        threshold
    sr: int
        sampling rate
    window: ndarray
        window function

    Return
    ----------
    filter: ndarray
        high pass filter
    """

    filter = np.zeros(sr)
    filter[freq:-freq] = 1

    # IFFT
    filter = np.fft.ifft(filter).real
    # Shift
    filter = np.roll(filter, filter.size//2)

    # multiply window function
    center = filter.size//2
    filter = filter[center-window.size//2:center+window.size//2]*window
    return filter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="Wave file name")
    parser.add_argument("--f_high", type=int, default=3000,
                        help="high frequency")
    parser.add_argument("--win_size", type=int, help="window size",
                        default=300)
    args = parser.parse_args()

    # load sound file
    data, sr = librosa.load(args.filename)

    # set window function
    window = signal.hamming(args.win_size)

    f = np.arange(0, sr//2+1)

    # High Pass Filter
    hpf = HPF(args.f_high, sr, window)
    _hpf = np.fft.rfft(hpf, sr)
    angles = np.unwrap(np.angle(_hpf))

    data_hpf=Convolution(data,hpf)

    # High Pass Filter Frequency Response
    fig1, ax1 = plt.subplots()
    ax1.plot(f, np.abs(_hpf))
    ax1.set(title="Filter Amplitude", xlabel="Frequency [Hz]",
            ylabel="Amplitude")
    fig1.savefig("FilterAmplitude.png")

    fig2, ax2 = plt.subplots()
    ax2.plot(f, angles)
    ax2.set(title="Filter Phase", xlabel="Frequency [Hz]",
            ylabel="Angle (Degrees)")
    ax2.grid()
    fig2.savefig("FilterPhase.png")

    #Sound file plot
    fig3=plt.figure()
    fig3.subplots_adjust(hspace=0.6)
    ax3_1=fig3.add_subplot(211)
    librosa.display.waveshow(data,sr=sr)
    ax3_1.set(title="Input Signal", xlabel="Time[sec]", ylabel="Magnitude")
    
    ax3_2=fig3.add_subplot(212)
    librosa.display.waveshow(data_hpf,sr=sr)
    ax3_2.set(title="High Pass Signal", xlabel="Time[sec]", ylabel="Magnitude")
    fig3.savefig("Signal.png")

    #Spectrogram
    spec1 = STFT(data)
    spec_db1 = librosa.amplitude_to_db(np.abs(spec1))

    spec2 = STFT(data_hpf)
    spec_db2 = librosa.amplitude_to_db(np.abs(spec2))
    
    fig4=plt.figure()
    fig4.subplots_adjust(hspace=0.6)
    ax4_1 = fig4.add_subplot(211)
    img1 = librosa.display.specshow(
        spec_db1, sr=sr, hop_length=512, x_axis="time", y_axis="linear")
    ax4_1.set(title="Spectrogram (Input Signal)", xlabel="Time[sec]", ylabel="Frequency[Hz]")
    fig4.colorbar(img1, ax=ax4_1)
    
    ax4_2=fig4.add_subplot(212)
    img2 = librosa.display.specshow(
        spec_db2, sr=sr, hop_length=512, x_axis="time", y_axis="linear")
    ax4_2.set(title="Spectrogram (Filtering Signal)", xlabel="Time[sec]", ylabel="Frequency[Hz]")
    fig4.colorbar(img2, ax=ax4_2)
    fig4.savefig("Spectrogram.png")

    soundfile.write("FilterdSound.wav",data_hpf,samplerate=sr)


if __name__ == "__main__":
    main()
