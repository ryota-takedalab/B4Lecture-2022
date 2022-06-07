import matplotlib.pyplot as plt
import librosa as lb
from mfcc import *
import argparse


def main():
    # Argparse
    parser = argparse.ArgumentParser(description='Name of the Audio File')
    parser.add_argument('-fn', metavar='-file', dest='filename', type=str,
                        help='Enter the Audio File Name', required=True)
    parser.add_argument('-wl', metavar='-s', default=256, dest='win_length', type=int,
                        help='Enter window Length', required=False)
    parser.add_argument('-f0', metavar='-f', default=700, dest='basis_freq', type=int,
                        help='Enter Basis Frequency', required=False)
    parser.add_argument('-k', metavar='-del', default=4, dest='delta_width', type=int,
                        help='Width of Delta', required=False)
    parser.add_argument('-n', metavar='-mel', default=20, dest='n_filterbank', type=int,
                        help='Number of filter bank', required=False)
    parser.add_argument('-dim', metavar='-d', default=12, dest='dim', type=int,
                        help='Dimension of MFCC', required=False)
    args = parser.parse_args()

    # Reading Data and sample rate from audio, then convert the sample rate to 16kHz and channel to mono
    audio, samplerate = lb.load("ex1.wav", sr=16000, mono=True)

    # Filter bank
    mfcc = Mfcc(data=audio, samplerate=samplerate, k=args.delta_width, f0=args.basis_freq,
                n=args.n_filterbank, win_length=args.win_length)
    f_bank = mfcc.melfilterbank()
    # display the filter bank
    plt.figure(figsize=(10, 4))
    freq = np.linspace(0, samplerate / 2000, f_bank.shape[1])
    for flt in f_bank:
        plt.plot(freq, flt)
    plt.title("mel filter bank")
    plt.xlim(0, samplerate/2000)
    plt.ylim(0, 1)
    plt.xlabel("Frequency[kHz]")
    plt.savefig('filterbank.png')
    plt.show()
    plt.close()

    # Calculate MFCC
    dim = args.dim
    dct = mfcc.mfcc(dim=args.dim)
    time = audio.size / samplerate
    max_time = time + (time / dct.shape[0])
    extent = [0, max_time, 0, dim]
    aspect = 4 * max_time / (dim * 10)
    delta = mfcc.del_mfcc(dct)
    delta_delta = mfcc.del_mfcc(delta)
    # plot spectrogram
    # Prepare the plots figures
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.45)

    values1, y, x, im1 = ax1.specgram(audio, Fs=samplerate, cmap='jet', aspect='auto')
    fig.colorbar(im1, ax=ax1)
    ax1.set_title("Spectrogram")
    ax1.set_xlabel("Time [sec]")
    ax1.set_ylabel("Frequency [Hz]")

    ax2.set(xlabel="Time [sec]", ylabel='MFCC', title='MFCC')
    im2 = ax2.imshow(dct.T, extent=extent, aspect='auto', origin='lower', cmap='jet')
    cbar = fig.colorbar(im2, ax=ax2)

    ax3.set(xlabel="Time [sec]", ylabel='MFCC', title='ΔMFCC')
    im3 = ax3.imshow(delta.T, extent=extent, aspect='auto', origin='lower', cmap='jet')
    cbar = fig.colorbar(im3, ax=ax3)

    ax4.set(xlabel="Time [sec]", ylabel='MFCC', title='ΔΔMFCC')
    im4 = ax4.imshow(delta_delta.T, extent=extent, aspect='auto', origin='lower', cmap='jet')
    cbar = fig.colorbar(im4, ax=ax4)
    plt.savefig('spectrogram.png')
    plt.show()


if __name__ == '__main__':
    main()
