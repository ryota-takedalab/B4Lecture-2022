import librosa as lb
import argparse
from spectrum_envelope import *
from f0_fundamental_freq import *


def main():
    # Argparse
    parser = argparse.ArgumentParser(description='Name of the Audio File')
    parser.add_argument('-fn', metavar='-f', dest='filename', type=str, help='Enter the Audio File Name',
                        required=True)
    parser.add_argument('-wl', metavar='-s', dest='window_length', type=int,
                        help='Enter Shift Length', default=1024, required=False)
    parser.add_argument('-sl', metavar='-s', dest='shift_length', type=int,
                        help='Enter Shift Length', default=512, required=False)
    parser.add_argument('-l', metavar='-lf', dest='lifter_coeff', type=int,
                        help='Enter Lifter Cutoff Freq', default=32, required=False)
    parser.add_argument('-ld', metavar='-lpc', dest='lpc_deg', type=int,
                        help='Enter LPC Degree', default=64, required=False)
    args = parser.parse_args()

    # Reading Data and sample rate from audio, then convert the sample rate to 16kHz and channel to mono
    audio, samplerate = lb.load(args.filename, sr=16000, mono=True)

    # Calculate and Plot Fundamental Frequency
    f0_plot(audio, window=args.window_length, N=args.shift_length, sr=16000, lifter=args.lifter_coeff)

    # Envelope
    window = 1024
    frame_env = int(len(audio)//10)
    win_data = audio[frame_env:frame_env+window] * np.hanning(window)
    spectrum_envelope(win_data, window=args.window_length, sr=16000, lpc_deg=args.lpc_deg)
    return


if __name__ == "__main__":
    main()
