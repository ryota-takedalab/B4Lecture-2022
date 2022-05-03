import numpy as np
import matplotlib.pyplot as plt
import librosa as lb
import soundfile as sf


def istft(spectrogram, nperseg=1024, noverlap=256):
    """Compute an inverse short-time Fourier transform (STFT) from spectrogram data to audio signal

        Args:
            spectrogram (np.ndarray): Data to be transformed.
            nperseg (int): Length of each segment. (Default value = 1024)
            noverlap (int): Number of points to overlap between segments. (Default value = 128)


        Returns:
            decibel (np.ndarray): The discrete short-time Fourier transform.
        """

    new_audio = np.zeros((spectrogram.shape[1]-1)*(nperseg - noverlap) + nperseg, dtype='complex_')
    for i in range(spectrogram.shape[1]):
        x_win = spectrogram.T[i]
        new_audio[i * (nperseg - noverlap):i * (nperseg - noverlap) + nperseg] += np.fft.ifft(x_win, axis=0)
    return new_audio.astype('float32')


# Make a function to Slice the frequency to 1024 Hz each for better performance
def stft(audio, nperseg=1024, noverlap=256):
    """Compute a discrete short-time Fourier transform (STFT) to an audio signal

    Args:
        audio (np.ndarray): Signal to be transformed
        nperseg (int): Length of each segment. (Default value = 1024)
        noverlap (int): Number of points to overlap between segments. (Default value = 128)


    Returns:
        decibel (np.ndarray): The discrete short-time Fourier transform
    """

    win = np.hanning(nperseg)
    audio_len = len(audio)
    step = np.ceil((audio_len - nperseg)/(nperseg - noverlap)).astype(int)
    spect = np.empty((0, nperseg))
    for m in range(step):
        # Apply windowing
        x_win = win * audio[m * (nperseg - noverlap):m * (nperseg - noverlap) + nperseg]
        # Apply FFT to the windowed signal and get the positive signal only
        spectrum = np.fft.fft(x_win)
        spect = np.append(spect, [spectrum], axis=0)
    return spect.T, nperseg


# Extra function to convert the transformed data (spect) into logarithmic value (decibel)
def db(spect):
    decibel = np.abs(spect)
    decibel = 20 * np.log10(decibel / np.max(decibel))
    return decibel


def main():
    # Reading Data and sample rate from audio, then convert the sample rate to 16kHz and channel to mono
    audio, samplerate = lb.load('exe1.wav', sr=16000, mono=True)

    # Number of data and length (seconds) of the audio
    num = audio.shape[0]
    length = num / samplerate
    print(f"audio length = {length}s")

    # STFT the data
    (spectrum, nperseg) = stft(audio)

    # ISTFT the spectrogram data
    spectrum2 = istft(spectrum)

    # Prepare the plots figures
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.45)

    # Subplot 1 (Frequency-time Domain plot for source input)
    ax[0].plot(np.arange(audio.shape[0]) / samplerate, audio)
    ax[0].set_xlabel('Time [s]')
    ax[0].set_ylabel('Amplitude [unknown]')
    ax[0].set_title('Input Audio Frequency-time Graph')

    # Subplot 2 (Spectrogram)
    # [:1024 // 2] is used so only the positive part is presented in spectrogram
    ax[1].imshow(db(spectrum[:nperseg // 2]), origin='lower', cmap='viridis',
                 extent=(0, length, 0, samplerate / 2 / 1000))
    ax[1].axis('tight')
    ax[1].set_ylabel('Frequency [kHz]')
    ax[1].set_xlabel('Time [s]')
    ax[1].set_title('Spectrogram')
    cbar_ax = fig.add_axes([0.93, 0.39, 0.01, 0.205])
    plt.colorbar(ax[1].images[0], orientation="vertical", cax=cbar_ax)

    # Subplot 3 (Frequency-time Domain plot for Re-synthesized signal)
    ax[2].plot(np.arange(spectrum2.shape[0]) / samplerate, spectrum2)
    ax[2].set_xlabel('Time [s]')
    ax[2].set_ylabel('Amplitude [unknown]')
    ax[2].set_title('Re-synthesized signal Frequency-time Graph')
    plt.show()
    fig.savefig('ex1_all_graph.png')
    plt.close()

    # Comparison between pre-built numpy function spectrogram to STFT made for this exercise
    fig2, ax2 = plt.subplots(2, 1, figsize=(10, 10))

    # Subplot 1 (Spectrogram used in Exercise)
    ax2[0].imshow(db(spectrum[:nperseg // 2]), origin='lower', cmap='viridis',
                  extent=(0, length, 0, samplerate / 2 / 1000))
    ax2[0].axis('tight')
    ax2[0].set_ylabel('Frequency [kHz]')
    ax2[0].set_xlabel('Time [s]')
    ax2[0].set_title('Spectrogram from Exercise Function')

    # Subplot 2 (Spectrogram from pre-built matplotlib.pyplot function)
    ax2[1].specgram(audio, Fs=samplerate)
    ax2[1].axis('tight')
    ax2[1].set_ylabel('Frequency [kHz]')
    ax2[1].set_xlabel('Time [s]')
    ax2[1].set_title('Spectrogram from Pre-built matplotlib.pyplot function')
    plt.show()
    fig2.savefig('ex1_spectrogram_comparison.png')
    plt.close()

    # Save a new synthesized audio
    sf.write('exe1_synthesized.wav', spectrum2, samplerate)


if __name__ == '__main__':
    main()
