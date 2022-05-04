import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


def load_file():
    file_path = "sample01.wav"
    data, rate = librosa.load(file_path)

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
    #window function
    win_fc = np.hamming(window)

    frame = (len(data)-window+step)//step
    spec = np.zeros([window//2+1, frame], dtype="complex64")
    for i in range(frame):
        #start position to cut signal
        start = step*i

        #cut signal times window function
        window_signal=data[start:start+window]*win_fc

        #fast fourier transform
        spec[:, i] = np.fft.rfft(window_signal)
    
    return spec


def ISTFT(spec):
    """
    Invert Short Time Fourier Transform

    Parameters
    ----------
    spec : Spectrogram data (np.array)
    """
    spec_t=np.transpose(spec)
    
    ispec=np.fft.irfft(spec_t)
    #print(ispec.shape)#(163,1024)
    window=ispec.shape[1]

    left=ispec[:,:window//2]
    right=ispec[:,window//2:]
    ispec=left
    ispec[1:]=ispec[1:]+right[:-1]
    #ispec = (left[0], left[1]+right[0],...,left[511]+right[510])
    #print(ispec.shape)#(163,512)

    resyn_signal=ispec.reshape(-1)
    return resyn_signal


def plot_function(win,rate,data,spec_db,resyn_data):
    """
    save graph as png
    """
    fig0 = plt.figure()
    ax0 = fig0.add_subplot(111)
    librosa.display.waveshow(data, sr=rate)
    ax0.set(title="Input Signal", xlabel="Time[sec]", ylabel="Magnitude")
    plt.savefig("InputSignal")

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    img = librosa.display.specshow(
        spec_db, sr=rate, hop_length=win//2, x_axis="time", y_axis="linear")
    ax1.set(title="Spectrogram", xlabel="Time[sec]", ylabel="Frequency[Hz]")
    fig1.colorbar(img, ax=ax1)
    plt.savefig("Spectrogram")

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    librosa.display.waveshow(resyn_data, sr=rate)
    ax2.set(title="Resynthesized Signal", xlabel="Time[sec]", ylabel="Magnitude")
    plt.savefig("ResynthesizedSignal")


def main():
    data, rate = load_file()

    win = 1024
    step = win//2

    spec = STFT(data, win, step)
    spec_db = librosa.amplitude_to_db(np.abs(spec))

    resyn_data=ISTFT(spec)

    plot_function(win,rate,data,spec_db, resyn_data)
    

if __name__ == "__main__":
    main()