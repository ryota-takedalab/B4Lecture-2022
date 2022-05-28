import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read


from my_functions import f0
from my_functions import stft
from my_functions import envelope

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ex3 f0-estimation')
    parser.add_argument("-i", "--input", help="input file")
    parser.add_argument('--mode', choices=['LP', 'HP'], default='LP')
    args = parser.parse_args()
    
    fig = plt.figure(figsize=(12, 8))
    fig.subplots_adjust(hspace=0.6)
    
    fs, wav = read(args.input)
    Zxx, t, f = stft.stft(wav, fs)
    
    # f0 estimation with cepstrum
    f0_estimated_cepstrum = f0.f0_estimate_cepstrum(wav, fs)
    ax1 = fig.add_subplot(221)
    ax1.set_title("f0(cepstrum)")
    im = ax1.imshow(
        20 * np.log10(np.abs(np.flipud(Zxx[:Zxx.shape[0] // 2]))),
        cmap=plt.cm.jet,
        aspect="auto",
        extent=[t[0], t[-1], f[0], f[len(f) // 2]])
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Frequency [Hz]")
    fig.colorbar(im, ax=ax1)
    ax1.plot(t, f0_estimated_cepstrum,
             label="f0", color="black", alpha=0.5)
    ax1.legend()

    # NOTE: C4: 261.626, E4: 329.628, G4: 391.995
    
    f0_estimated_autocorrelation = f0.f0_estimate_autocorrelation(wav, fs)
    ax2 = fig.add_subplot(222)
    ax2.set_title("f0(auto-correlation)")
    im = ax2.imshow(
        20 * np.log10(np.abs(np.flipud(Zxx[:Zxx.shape[0] // 2]))),
        cmap=plt.cm.jet,
        aspect="auto",
        extent=[t[0], t[-1], f[0], f[len(f) // 2]])
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Frequency [Hz]")
    fig.colorbar(im, ax=ax2)
    ax2.plot(t, f0_estimated_autocorrelation,
             label="f0", color="black", alpha=0.5)
    ax2.legend()

    # envelope
    frame_length = 512
    target_start_frame = len(wav) // 4
    target_frame = wav[target_start_frame: target_start_frame + frame_length]
    log_spectrum_wav = envelope.log_spectrum(target_frame)
    f = np.linspace(0, fs // 2, frame_length // 2)
    
    # envelope based on cepstrum
    envelope_cepstrum_wav = envelope.envelope_cepstrum(target_frame)
    ax3 = fig.add_subplot(223)
    ax3.set_title("envelope based on cepstrum")
    ax3.set_xlabel("Frequency [Hz]")
    ax3.set_ylabel("Amplitude [dB]")
    ax3.plot(f, log_spectrum_wav[:frame_length // 2],
             label="log amplitude")
    ax3.plot(f, envelope_cepstrum_wav[0: frame_length // 2],
             label="sprctrum envelope")
    ax3.legend()
    
    # envelope based on LPC
    p = 32
    envelope_lpc_wav = envelope.envelope_lpc(target_frame, p, fs)
    ax4 = fig.add_subplot(224)
    ax4.set_title("envelope based on LPC")
    ax4.set_xlabel("Frequency [Hz]")
    ax4.set_ylabel("Amplitude [dB]")
    ax4.plot(f, log_spectrum_wav[:frame_length // 2],
             label="log amplitude")
    ax4.plot(f, envelope_lpc_wav[0: frame_length // 2],
             label="sprctrum envelope")
    ax4.legend()
    
    plt.show()
    fig.savefig("result.png")
