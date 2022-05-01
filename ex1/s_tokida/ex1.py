import soundfile as sf
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def STFT(data, Fs, Fl, overlap): 

    shift = int((Fl-overlap)//(Fs-overlap))  # number of shifts
    win = np.hamming(Fs)  # humming window fuction
    spec = []
    pos = 0

    for i in range(shift):
        shift_data = data[int(pos):int(pos+Fs)]
        if len(shift_data) == Fs:
            windowed_data = win * shift_data
            fft_data = np.fft.fft(windowed_data)  # fft
            spec.append(np.real(fft_data))
            pos += (Fs-overlap)

    return spec

def ISTFT(spec, Fs, Fl, overlap):
    ispec = np.zeros(Fl)
    pos = 0

    for i in range(len(spec)):
        ifft_data = np.fft.ifft(spec[i])
        ispec[int(pos):int(pos+Fs)] += np.real(ifft_data)
        pos += (Fs-overlap)

    return ispec

def main():
    # load file
    filepath = 'ex1.wav'
    data, samplerate = sf.read(filepath)
    # culculate
    Fs = 1024  # Frame size [2048, 1024, 128]
    Fl = data.shape[0]  # Frame length  # len(data) = 301644
    overlap = Fs * 0.5

    #STFT
    spec = STFT(data, Fs, Fl, overlap)

    #ISTFT
    ispec = ISTFT(spec, Fs, Fl, overlap)

    #PLOT
    t = np.arange(0, Fl)/samplerate  # time bar

    fig = plt.figure(figsize=(8,6)) 
    ax1 = fig.add_subplot(311)  # Original signal
    ax1.plot(t, data)
    ax1.set_ylabel('Magnitude')
    ax1.set_title('Original signal')

    ax2 = fig.add_subplot(312)  # Spectrogram
    spec_log = 20*np.log10(np.abs(spec))
    im = ax2.imshow(spec_log.T, extent=[0, float(Fl)/samplerate, 0, samplerate], aspect='auto')
    divider_ax2 = make_axes_locatable(ax2)
    cax2 = divider_ax2.append_axes('right', size='2%', pad=0.1)
    cbar = fig.colorbar(im, cax=cax2)  # color bar
    ax2.set_ylabel('Frequency[Hz]')
    ax2.set_title('Spectrogram')

    ax3 = fig.add_subplot(313)  # Re-synthesized signal
    ax3.plot(t, ispec)
    ax3.set_xlabel('Time[s]')
    ax3.set_ylabel('Magnitude')
    ax3.set_title('Re-synthesized signal')

    fig.tight_layout()

    # align x-axis
    fig.canvas.draw()
    axpos1 = ax1.get_position()
    axpos2 = ax2.get_position()
    axpos3 = ax3.get_position()
    ax1.set_position([axpos1.x0, axpos1.y0, axpos2.width, axpos1.height])
    ax3.set_position([axpos3.x0, axpos3.y0, axpos2.width, axpos3.height])

    plt.show()
    plt.close()
    fig.savefig('ex1_tokida_1024.png')

if __name__ == "__main__":
    main()