import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import argparse


def load_file(filename):
    data, rate = librosa.load(filename, sr=None)
    # print(rate)#44100

    return data, rate


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
    spec = np.zeros([window//2+1, frame], dtype="complex64")

    # use "for"
    # -----------
    """
    for i in range(frame):
        #start position to cut signal
        start = step*i

        #cut signal times window function
        window_signal=data[start:start+window]*win_fc

        #fast fourier transform
        spec[:, i] = np.fft.rfft(window_signal)
    """
    # -----------
    # not use "for"
    # -----------
    framed_signal = np.lib.stride_tricks.as_strided(data, shape=(
        window, frame), strides=(data.strides[0], data.strides[0]*step))

    # win_fc.shape=(window,) -> win_fc[:,np.newaxis].shape=(window,1)
    window_signal = framed_signal*win_fc[:, np.newaxis]
    # fast fourier transform
    spec = np.fft.rfft(window_signal, axis=0)
    # -----------

    return spec


def ISTFT(spec):
    """
    Invert Short Time Fourier Transform

    Parameters
    ----------
    spec : Spectrogram data (np.array)
    """

    spec_t = np.transpose(spec)

    ispec = np.fft.irfft(spec_t)
    # print(ispec.shape)#(163,1024)
    window = ispec.shape[1]
    
    left = ispec[:, :window//2]
    right = ispec[:, window//2:]
    ispec = left
    ispec[1:] = ispec[1:]+right[:-1]
    #ispec = (left[0], left[1]+right[0],...,left[511]+right[510])
    # print(ispec.shape)#(163,512)

    resyn_signal = ispec.reshape(-1)

    return resyn_signal


def plot_function(win, rate, data, spec_db, resyn_data):
    """
    save graph as png
    """
    fig = plt.figure(figsize=(10,8))

    #Input Signal
    ax0 = fig.add_subplot(311)
    librosa.display.waveshow(data, sr=rate)
    ax0.set(title="Input Signal", xlabel="Time[sec]", ylabel="Magnitude")
    ax0.set_position([0.1,0.7,0.68,0.2])

    #Spectrogram
    ax1 = fig.add_subplot(312)
    img = librosa.display.specshow(
        spec_db, sr=rate, hop_length=win//2, x_axis="time", y_axis="linear")
    ax1.set(title="Spectrogram", xlabel="Time[sec]", ylabel="Frequency[Hz]")
    ax1.set_position([0.1,0.3,1,0.25])
    fig.colorbar(img, ax=ax1)

    #Resynthesized Signal
    ax2 = fig.add_subplot(313)
    librosa.display.waveshow(resyn_data, sr=rate)
    ax2.set(title="Resynthesized Signal",
            xlabel="Time[sec]", ylabel="Magnitude")
    ax2.set_position([0.1,0.1,0.68,0.2])

    plt.savefig("Result")


def main():
    parser=argparse.ArgumentParser()

    parser.add_argument("filename",help="Wave File Name")
    parser.add_argument("--win",default=1024,help="Window")
    parser.add_argument("--step",default=512,help="step")

    args=parser.parse_args()

    data, rate = load_file(args.filename)

    spec = STFT(data, args.win, args.step)
    spec_db = librosa.amplitude_to_db(np.abs(spec))

    resyn_data = ISTFT(spec)

    plot_function(args.win, rate, data, spec_db, resyn_data)


if __name__ == "__main__":
    main()
