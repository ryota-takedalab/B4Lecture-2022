import numpy as np
import librosa
import librosa.display
import os
import my_function
import matplotlib.pyplot as plt
import soundfile as sf


def main():
    # load audio file
    # get current working directory
    dir = os.path.dirname(os.path.abspath(__file__)) + "/"
    audio_path = dir + "ex2_sample.wav"
    # get waveform data and sample rate
    wav, sr = librosa.load(audio_path, mono=True)

    # parameter
    hop = 0.5
    win_length = 1024
    hop_length = int(win_length * hop)
    f_low = 1000
    fir_size = 1000

    # create low-pass filter
    lpf = my_function.LowPassFilter(f_low=f_low, sr=sr, N=fir_size)

    # analize filter
    lpf_freq = np.fft.rfft(lpf, sr)
    lpf_amp = np.abs(lpf_freq)
    lpf_phase = np.unwrap(np.angle(lpf_freq))

    # plot　the frequency response
    plt.rcParams["figure.figsize"] = (10, 14)
    fig, ax = plt.subplots(2, 1)
    fig.subplots_adjust(hspace=0.5)

    ax[0].plot(lpf_amp)
    ax[0].set(title="Filter amplitude", xlabel="Frequency[Hz]", ylabel="Amplitude")
    ax[0].grid()

    ax[1].plot(lpf_phase)
    ax[1].set(title="Filter phase", xlabel="Frequency[Hz]", ylabel="Phase[rad]")
    ax[1].grid()

    # plot waveform and spectrogram
    fig2, ax2 = plt.subplots(4, 1)
    fig2.subplots_adjust(hspace=0.5)

    # apply low-pass filter
    flt_wav = my_function.convolve(wav, lpf)

    # stft
    original_amp = my_function.stft(wav, hop, win_length)
    filtered_amp = my_function.stft(flt_wav, hop, win_length)

    # convert an amplitude spectrogram to dB_scaled spectrogram
    original_db = librosa.amplitude_to_db(np.abs(original_amp))
    filtered_db = librosa.amplitude_to_db(np.abs(filtered_amp))

    # draw original signal
    librosa.display.waveshow(wav, sr=sr, color="b", ax=ax2[0])
    ax2[0].set(title="Original signal", xlabel=None, ylabel="Magnitude")

    # draw fitered signal
    librosa.display.waveshow(flt_wav, sr=sr, color="b", ax=ax2[1])
    ax2[1].set(title="Filtered signal", xlabel=None, ylabel="Magnitude")

    # draw original spectrogram (log scale)
    img = librosa.display.specshow(
        original_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="log",
        ax=ax2[2],
        cmap="plasma",
    )
    ax2[2].set(title="Original Spectrogram", xlabel=None, ylabel="Frequency [Hz]")
    ax2[2].set_yticks([0, 128, 512, 2048, 8192])
    fig.colorbar(img, aspect=10, pad=0.01, extend="both", ax=ax2[2], format="%+2.f dB")

    # draw filtered spectrogram (log scale)
    img = librosa.display.specshow(
        filtered_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="log",
        ax=ax2[3],
        cmap="plasma",
    )
    ax2[3].set(title="Filtered Spectrogram", xlabel=None, ylabel="Frequency [Hz]")
    ax2[3].set_yticks([0, 128, 512, 2048, 8192])
    fig.colorbar(img, aspect=10, pad=0.01, extend="both", ax=ax2[3], format="%+2.f dB")

    # graph adjustment
    ax2_pos_0 = ax2[0].get_position()
    ax2_pos_1 = ax2[1].get_position()
    ax2_pos_2 = ax2[2].get_position()
    ax2[0].set_position([ax2_pos_0.x0, ax2_pos_0.y0, ax2_pos_2.width, ax2_pos_2.height])
    ax2[1].set_position([ax2_pos_1.x0, ax2_pos_1.y0, ax2_pos_2.width, ax2_pos_2.height])

    # save and show figure of result
    fig.savefig(dir + "ex2_responce_lpf1000Hz.png")
    fig2.savefig(dir + "ex2_spectrogram_lpf1000Hz.png")
    plt.show()

    # export the data processed by the low-pass filter.
    sf.write("ex2_lpf1000Hz.wav", flt_wav, sr, subtype="PCM_24")


if __name__ == "__main__":
    main()
