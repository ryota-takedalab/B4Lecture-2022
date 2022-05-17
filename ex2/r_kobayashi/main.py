# include flake8, black

import argparse

import numpy as np
import librosa
import soundfile as sf
import sys
import os
import warnings

from stft import stft
import filters
import mkfig

# ignore warnings
warnings.filterwarnings("ignore")

def conv1d(in1, in2, mode = "full"):
    """
    Convolve two 1-dimensional arrays.
    Convolve in1 and 1n2 with output size determined by mode argument.

    Parameters:
        in1 : array_like
            First input.
        in2 : array_like
            Second input. Should have the same number of dimension as in1.
        mode : str {'full', 'valid', 'same'}, optional
            A string indicating the size of the output:

                full
                The output is the full discrete linear convolution of the inputs. (Default)

                valid
                The output consists only of those elements that do not rely on the zero-padding.
                In 'valid' mode, either in1 or in2 must be at least as large as the other in every dimension.

                same
                The output is the same size as in1, centerd with respect to the 'full'output.

    Returns:
        out : ndarray
            A 1-dimensional array containing a subset of the discrete linear convolution of in1 with in2.
    """

    in1_len = len(in1)
    in2_len = len(in2)

    assert in1_len > in2_len, "invalid ndarray size in conv1d function"

    out = np.zeros(in1_len + in2_len)
    for i in range(in2_len):
        out[i : i + in1_len] += in1 * in2[i]
    
    if mode == "full":
        out = out[0 : in1_len + in2_len -1]
    elif mode == "valid":
        out = out[in2_len - 1 : in1_len]
    elif mode == "same":
        out = out[0 : in1_len]
    else:
        print("error: unavailable mode in func conv1d")
        sys.exit(1)

    return out

def main(args):
    # load audio file
    path = os.path.dirname(__file__)
    wav, sr = librosa.load(path + args.sc, mono = True)

    # parameter
    hop = 0.5
    win_length = 1024
    hop_length = int(win_length * hop)
    fir_size = 200

    # create filter
    if args.hpf:
        fir = filters.HPF(sr, args.hpf[0], fir_size)
    elif args.lpf:
        fir = filters.LPF(sr, args.lpf[0], fir_size)
    elif args.bpf:
        fir = filters.BPF(sr, args.bpf[0], args.bpf[1], fir_size)
    elif args.bef:
        fir = filters.BEF(sr, args.bef[0], args.bef[1], fir_size)
    
    # analize filter
    fir_fft = np.fft.fft(fir)
    amp = np.abs(fir_fft)
    amp_db = librosa.amplitude_to_db(np.abs(amp))
    phase = np.unwrap(np.angle(fir_fft))
    freq = np.arange(0, sr / 2, (sr //2) / (fir_size // 2 + 1))

    # show and save a figure of filter characteristic (amplitude and phase)
    fig = mkfig.filterchar_show(freq, amp_db, phase, fir_size)
    fig.savefig(path + "./result\\filterchar.png")

    # spectrogram of original wav data
    spec1 = stft(wav, hop = hop, win_length = win_length)
    db1 = librosa.amplitude_to_db(np.abs(spec1))

    # apply filter to wav data
    wav_filtered = conv1d(wav, fir)
    #spectrogram of filterd wav data
    spec2 = stft(wav_filtered, hop = hop, win_length = win_length)
    db2 = librosa.amplitude_to_db(np.abs(spec2))

    # show and save both spectrograms
    fig = mkfig.double_specshow(
        db1,
        db2,
        sr = sr,
        hop_length = hop_length,
        x_axis = "time",
        y_axis = "linear",
        title1 = "Original spectrogram",
        title2 = "Filtered spectrogram"
    )
    fig.savefig(path + "./result\\spectrogram.png")

    #save filterd wav data
    sf.write(path + args.dst, wav_filtered, sr)


if __name__ == "__main__":
    # process args
    parser = argparse.ArgumentParser(
        description = "apply filter (HPF, LPF, BPF, BEF) to wav file"
    )
    parser.add_argument(
        "sc",
        type = str,
        default = "/wav/sample.wav",
        help = "input filename with extension (Default : wav/sampe.wav",
    )
    parser.add_argument(
        "dst",
        type = str,
        default = "/result/sample_filtered.wav",
        help = "output filename with extension (Default : result/sample_filtered.wav",
    )
    parser.add_argument(
        "--hpf",
        type = int,
        nargs = 1,
        metavar = "freq",
        help = "cutoff frequency [Hz]",
    )
    parser.add_argument(
        "--lpf",
        type=int,
        nargs=1,
        metavar="freq",
        help="cutoff frequency [Hz]",
    )
    parser.add_argument(
        "--bpf",
        type = int,
        nargs = 2,
        metavar = "freq",
        help = "low and high frequency [Hz]",
    )
    parser.add_argument(
        "--bef",
        type = int,
        nargs = 2,
        metavar = "freq",
        help = "low and high frequency [Hz]",
    )

    args = parser.parse_args()

    num_filter = (
        (args.hpf != None)
        + (args.lpf != None)
        + (args.bpf != None)
        + (args.bef != None)
    )

    if num_filter == 0:
        parser.error("One of filters (HPF, LPF, BPF, BEF) must be set.")
    elif num_filter > 1:
        parser.error("multiole filters can't be set.")

    main(args)