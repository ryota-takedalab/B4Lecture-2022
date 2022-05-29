import numpy as np
import librosa
import librosa.display
from matplotlib import pyplot as plt
import argparse

import pyworld as pw


def stft(data, window=1024, step=512):
    """Short Time Fourier Transform

    Args:
        data (ndarray): input signal
        window (int, optional): Length of window. Defaults to 1024.
        step (int, optional): Length of shift. Defaults to 512.

    Returns:
        ndarray: spectrogram
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


def get_cut_signal(input, win_size, shift_size):
    """cut input signal

    Args:
        input (ndarray): input signal
        win_size (int): window size
        shift_size (int): shift length

    Returns:
        ndarray: framed signal
    """
    length = input.shape[0]
    frame_num = (length-win_size)//shift_size+1
    frames = np.zeros((frame_num, win_size)+input.shape[1:])

    for i in range(frame_num):
        # start position to cut signal
        start = shift_size*i
        frames[i] = input[start:start+win_size]

    return frames


def auto_correlation(input, win_size):
    """auto correlation

    Args:
        input (ndarray): input signal
        win_size (int): window size

    Returns:
        ndarray: auto correlation signal
    """
    frames = get_cut_signal(input, win_size, win_size//2)
    spec = np.fft.rfft(frames, win_size * 2)
    power = spec * spec.conj()
    ac = np.fft.irfft(power)

    ac = np.transpose(ac)
    ac = ac[:win_size]

    return ac


def detect_peak(input, neighbor):
    """detect peak from input signal

    Args:
        input (ndarray): input signal
        neighbor (int): detect range

    Returns:
        ndarray: peaks list
    """
    frames = get_cut_signal(input, 2*neighbor+1, 1)
    candidate = np.copy(input[neighbor:-neighbor])

    candidate[np.argmax(frames, axis=1) != neighbor] = -np.inf
    # calculate maximum of peak
    peaks = np.argmax(candidate, axis=0) + neighbor
    return peaks


def calc_f0_ac(ac, sr):
    """get F0 by AC

    Args:
        ac (ndarray): input AC
        sr (int): sampling rate

    Returns:
        ndarray: F0 list
    """
    peaks = detect_peak(ac, 10)
    f0 = sr/peaks

    f0[ac[0] < 0.25] = 0
    return f0


def ceps_f0(spec_db, sr, threshold):
    """get F0 by Cepstrum

    Args:
        spec_db (ndarray): spectrum signal (dB)
        sr (int): sampling rate
        threshold (int): threshold lengh

    Returns:
        ndarray, ndarray: F0, envelope
    """
    fsize = spec_db.shape[0]
    spec_db = np.concatenate([spec_db[-2:0:-1], spec_db], axis=0)

    ceps = np.fft.rfft(spec_db, axis=0).real

    env = np.zeros_like(ceps)
    env[:threshold] = ceps[:threshold]

    peaks = detect_peak(ceps[threshold:], 10)+threshold
    f0 = sr/peaks

    f0[ceps[0] < -32000] = 0

    env = np.fft.irfft(env, axis=0)[-fsize:]

    return f0, env


def levinson(r):
    """levinson durbin algorithm

    Args:
        r (ndarray): input data

    Returns:
        ndarray, ndarray: 
    """
    a = np.zeros_like(r)
    a[0] = 1.0
    sigma = r[0]
    for p in range(1, a.shape[0]):
        w = np.sum(a[:p]*r[p:0:-1], axis=0)
        k = w/sigma
        sigma = sigma-k*w

        a[1:p+1] = a[1:p+1]-k*a[p-1::-1]

    e = np.sqrt(sigma)

    return a, e


def lpc(signal, win_size, deg):
    """LPC algorithm

    Args:
        signal (ndarray): input signal
        win_size (int): window size
        deg (int): degree

    Returns:
        ndarray: envelope
    """
    ac = auto_correlation(signal, win_size)
    r = ac[:deg]

    a,e=levinson(r)

    env = e/np.fft.rfft(a, win_size, axis=0)
    env = librosa.amplitude_to_db(np.abs(env))

    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="Wave file name")
    parser.add_argument(
        "--f0", choices=["ac", "ceps"], default="ac", help="method for calculate f0")
    args = parser.parse_args()

    # load sound file
    data, sr = librosa.load(args.filename, sr=None, dtype="float", mono=True)
    win_size = 1024

    spec = stft(data)
    spec_db = librosa.amplitude_to_db(np.abs(spec))

    if args.f0 == "ac":
        ac = auto_correlation(data, win_size)
        f0 = calc_f0_ac(ac, sr)
    elif args.f0 == "ceps":
        f0 = ceps_f0(spec_db, sr, 68)[0]

    # _f0,_time=pw.dio(data,sr)
    # f0=pw.stonemask(data,_f0,_time,sr)

    librosa.display.specshow(spec_db, sr=sr, hop_length=win_size //
                             2, x_axis="time", y_axis="linear", cmap="rainbow")
    t = np.linspace(0, (data.size-win_size)/sr, f0.size)
    plt.plot(t, f0, color="black")
    plt.ylim(0, 1000)
    plt.title(f"F0 ({args.f0})")
    plt.xlabel("Time[sec]")
    plt.ylabel("Frequency[Hz]")
    plt.colorbar()
    plt.show()

    env1 = lpc(data, win_size, 32)
    env2 = ceps_f0(spec_db, sr, 68)[1]

    w = np.linspace(0, sr/2, win_size//2+1)
    plt.plot(w, spec_db[:, 135], label="original")
    plt.plot(w, env1[:, 135], label="lpc")
    plt.plot(w, env2[:, 135], label="cepstrum")
    plt.title("Envelope")
    plt.xlabel("Frequency[Hz]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
