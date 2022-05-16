import numpy as np
import soundfile as sf
from matplotlib import pyplot as plt
from math import pi

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
            spec.append(fft_data)
            pos += (Fs-overlap)

    return spec
    
def myconvolve(x, h):
    out = np.zeros(len(x) + len(h))
    for n in range(len(h)):
        out[n: n+len(x)] += x * h[n]
    return out[:len(x)]

def sinc(x):
    if x == 0:
        return 1.0
    else:
        return np.sin(x)/x

def BEF(f_low, f_high, samplerate, f_size):
    if f_size % 2 != 0:
        f_size += 1
    
    w_low = 2 * pi * f_low / samplerate
    w_high = 2 * pi * f_high / samplerate

    fir = []
    for n in range(-f_size // 2, f_size // 2 + 1):
        BEF_impulse = sinc(pi * n) - (w_high * sinc(w_high * n) + w_low * sinc(w_low * n)) / pi
        fir.append(BEF_impulse)

    fir = np.array(fir)
    window = np.hamming(f_size + 1)

    return fir * window

def myspectrogram(data, Fs, Fl, overlap, samplerate, title_name):
    spec = STFT(data, Fs, Fl, overlap)

    spec_log = 20*np.log10(np.abs(spec))
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    im = ax1.imshow(spec_log[:, :Fs//2].T, extent=[0, float(Fl)/samplerate, 0, samplerate/2], aspect='auto', origin = 'lower', cmap = 'hsv')
    fig.colorbar(im)
    ax1.set_xlabel('Time[s]')
    ax1.set_ylabel('Frequency[Hz]')
    ax1.set_title(title_name)

    plt.savefig('{}.png'.format(title_name))
    plt.show()
    plt.close()

def main():
    # load file
    filepath = 'ex1.wav'
    data, samplerate = sf.read(filepath)  # samplerate: 48000
    # culculate
    Fs = 1024  # Frame size
    Fl = data.shape[0]  # Frame length  # len(data) = 301644
    overlap = Fs * 0.5
    f_low = 3000
    f_high = 8000
    f_size = 100

    bef = BEF(f_low, f_high, samplerate, f_size)
    # frequency and phase characteristic of BEF filter
    freq_char = np.abs(np.fft.fft(bef))[:len(bef)//2+1]
    phase_char = np.unwrap(np.angle(np.fft.fft(bef))[:len(bef)//2+1]) * 180 / pi

    # convolve x and h
    data_h = myconvolve(data, bef)
    
    # plot filter characteristics
    t = np.linspace(0, samplerate/2, len(freq_char))
    plt.figure()
    plt.subplot(211)
    plt.plot(t, freq_char)
    plt.xlabel('Frequency[Hz]')
    plt.ylabel('Amptitude')
    plt.title('BEF Amptitude')

    t = np.linspace(0, samplerate/2, len(phase_char))
    plt.subplot(212)
    plt.plot(t, phase_char)
    plt.xlabel('Frequency[Hz]')
    plt.ylabel('Phase[rad]')
    plt.title('BEF Phase')
    plt.tight_layout()
    plt.savefig('filter_characteristic.png')
    plt.show()
    plt.close()

    #plot sound wav
    t = np.arange(0, Fl)/samplerate  # time bar
    plt.figure()
    plt.subplot(211)
    plt.plot(t, data)
    plt.xlabel('Time[s]')
    plt.ylabel('Frequency[Hz]')
    plt.title('Original signal')

    plt.subplot(212)
    plt.plot(t, data_h)
    plt.xlabel('Time[s]')
    plt.ylabel('Frequency[Hz]')
    plt.title('Filtered signal by BEF')

    plt.tight_layout()
    plt.savefig('wav_plot.png')
    plt.show()
    plt.close()
    
    #spectrogram by ex_1
    myspectrogram(data, Fs, Fl, overlap, samplerate, 'Original Spectrogram')
    myspectrogram(data_h, Fs, Fl, overlap, samplerate, 'Filtered Spectrogram')

    sf.write(file='ex2_filtered.wav', data=data_h, samplerate=samplerate)  # wavFile by ISTFT

if __name__ == "__main__":
    main()