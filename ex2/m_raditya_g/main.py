import argparse
import numpy as np
import matplotlib.pyplot as plt
import librosa as lb
import soundfile as sf
import time


def conv(v1, v2):
    """Return discrete, linear convolution of two one-dimensional vectors
        Equation based on numpy.convolve function mode='full'

        Args:
            v1 (np.ndarray): 1st 1D-array.
            v2 (np.ndarray): 2nd 1D-array.

        Returns:
            linconv (np.ndarray): The inverse short-time Fourier transform.
        """
    linconv = np.zeros(len(v2)+len(v1)-1)
    nmax = max(v1, v2, key=len)
    nmin = np.flipud(min(v2, v1, key=len))
    extrapolate = tuple(np.zeros(len(nmin)-1))
    # Extrapolate undefined to 0
    nmax = np.insert(nmax, 0, extrapolate)  # Is there a better way to extrapolate the undefined to 0?
    nmax = np.append(nmax, extrapolate)
    for n in range(len(nmax)-len(nmin)+1):
        linconv[n] += np.sum(nmin*nmax[n:n+len(nmin)])
    return linconv


def BPF(sr, fl=600, fh=3000, n=100):
    """Bandpass Butterworth Filter based on scipy.signal.butter

        Args:
            sr (int): Sampling rate
            fl (int): Low frequency cutoff. (Default value = 600)
            fh (int): High frequency cutoff. (Default value = 3000)
            n (int): Number of frequency filter (filter size). (Default value = 100)


        Returns:
            bpf_res (np.ndarray): Filtered Frequency.
            n (int): Number of frequency filter (filter size).
        """
    f1 = 2*np.pi*fl/sr
    f2 = 2*np.pi*fh/sr
    arange = np.arange(-n//2, n//2 + 1)
    bpf = (f2 * np.sinc(f2 * arange/np.pi) - f1 * np.sinc(f1 * arange/np.pi))/np.pi
    bpf_res = bpf*np.hamming(n+1)
    return bpf_res, n


def main():
    start = time.time()

    # Argparse
    parser = argparse.ArgumentParser(description='Name of the Audio File')
    parser.add_argument('-fn', metavar='-f', dest='filename', type=str, help='Enter the Audio File Name',
                        required=True)
    parser.add_argument('-f1', metavar='-1', dest='frequency_cut1', type=int, help='Enter Frequency Cut low',
                        required=True)
    parser.add_argument('-f2', metavar='-2', dest='frequency_cut2', type=int, help='Enter Frequency Cut high',
                        required=True)
    parser.add_argument('-fl', metavar='-filter', dest='n_filter', type=int, help='Enter Number of Filter Coefficient',
                        required=True)
    args = parser.parse_args()

    # Reading Data and sample rate from audio, then convert the sample rate to 16kHz and channel to mono
    audio, samplerate = lb.load(args.filename, sr=16000, mono=True)

    # Filtering the audio data and do convolution
    (bpf, n) = BPF(samplerate, fl=args.frequency_cut1, fh=args.frequency_cut2, n=args.n_filter)
    filtered_data = conv(audio, bpf)

    # BPF Filter amplitude and phase
    bpf_fft = abs(np.fft.fft(bpf))
    amp = 20*np.log10(bpf_fft)
    phase = np.unwrap(np.angle(bpf_fft)) * 180/np.pi
    frequency_lab = np.arange(0, samplerate/2, (samplerate//2)/(n//2+1))

    # Performance check
    print(time.time() - start)

    # BPF Amplitude Plot
    fig1, ax1 = plt.subplots(2, 1, figsize=(10, 10))
    ax1[0].plot(frequency_lab, amp[0: n//2+1])
    ax1[0].set_title("BPF Amplitude Characteristic")
    ax1[0].set_ylabel("Amplitude [dB]")
    ax1[0].set_xlabel("Frequency [Hz]")
    ax1[0].grid()

    # BPF Phase Plot
    ax1[1].plot(frequency_lab, phase[0: n//2+1])
    ax1[1].set_title("BPF Phase Characteristic")
    ax1[1].set_ylabel("Phase [deg]")
    ax1[1].set_xlabel("Frequency [Hz]")
    ax1[1].grid()
    fig1.savefig(f"ex2_filter_characteristic_{args.frequency_cut1}_{args.frequency_cut2}.png")

    # Prepare the plots figures
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(20, 10))
    plt.subplots_adjust(hspace=0.45)

    # Subplot 1 (Spectrogram used in exercise)
    values1, y, x, im1 = ax2.specgram(audio, Fs=samplerate)
    ax2.axis('tight')
    ax2.set_ylabel('Frequency [kHz]')
    ax2.set_xlabel('Time [s]')
    ax2.set_title('Unfiltered Spectrogram')
    fig.colorbar(im1, ax=ax2)

    # Subplot 2 (Spectrogram used in exercise)
    values, y, x, im = ax1.specgram(filtered_data, Fs=samplerate)
    ax1.axis('tight')
    ax1.set_ylabel('Frequency [kHz]')
    ax1.set_xlabel('Time [s]')
    ax1.set_title('Filtered Spectrogram')
    fig.colorbar(im, ax=ax1)
    plt.show()
    fig.savefig(f'ex2_spectrogram_comparison_{args.frequency_cut1}_{args.frequency_cut2}.png')
    plt.close()

    # Prepare the plots figures
    fig2, ax2 = plt.subplots(3, 1, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.45)

    # Subplot 1 (Audio Comparison)
    ax2[0].plot(np.arange(audio.shape[0]) / samplerate, audio)
    ax2[0].plot(np.arange(filtered_data.shape[0]) / samplerate, filtered_data)
    ax2[0].set_xlabel('Time [s]')
    ax2[0].set_ylabel('Amplitude [unknown]')
    ax2[0].set_title('Input Audio Frequency-time Graph Comparison (Filtered vs Unfiltered)')

    # Subplot 2 (Original Audio)
    ax2[1].plot(np.arange(audio.shape[0]) / samplerate, audio)
    ax2[1].set_xlabel('Time [s]')
    ax2[1].set_ylabel('Amplitude [unknown]')
    ax2[1].set_title('Input Audio Amplitude-time Graph (Unfiltered)')

    # Subplot 3 (Filtered Audio)
    ax2[2].plot(np.arange(filtered_data.shape[0]) / samplerate, filtered_data)
    ax2[2].set_xlabel('Time [s]')
    ax2[2].set_ylabel('Amplitude [unknown]')
    ax2[2].set_title('Input Audio Frequency-time Graph (Filtered)')
    plt.show()
    fig2.savefig(f'ex2_Amplitude_time_comparison_{args.frequency_cut1}_{args.frequency_cut2}.png')
    plt.close()

    # Save a new synthesized audio
    sf.write(f'ex2_synthesized_{args.frequency_cut1}_{args.frequency_cut2}.wav', filtered_data, samplerate)


if __name__ == '__main__':
    main()
