import numpy as np
import argparse
import librosa

from k_means import KMeans
from mfcc import MFCC


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="data file .csv or sound file")
    parser.add_argument("--k", type=int, help="cluster number")

    args = parser.parse_args()

    # load file
    if ".wav" in args.filename:
        # load sound file .wav
        data, sr = librosa.load(args.filename, sr=None, dtype="float", mono=True)
        win_size = 1024
        step = win_size // 2
        n_channels = 20
        f0 = 700
        mfcc = MFCC(data, sr, win_size, step, n_channels, f0)
        mfcc.mfcc_plot(args.filename)
    else:
        # load text from .csv
        data = np.loadtxt(args.filename + ".csv", delimiter=",", skiprows=1)

        if args.k == None:
            print("set cluster number: --k int")
            return
        model = KMeans(args.k, args.filename)
        model.fit(data)


if __name__ == "__main__":
    main()
