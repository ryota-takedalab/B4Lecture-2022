import numpy as np

def stft(wav, hop, win_length):
    hop_length = int(win_length * hop)
    window = np.hamming(win_length)
    spec = []
    for j in range(0, len(wav), hop_length):
        x = wav[j:j + win_length]
        #print("len(x):", len(x))
        if win_length > len(x):
            break
        x = window * x
        x = np.fft.fft(x)
        #print("fft(x).shape:", x.shape)
        spec.append(x)
    spec = np.array(spec).T
    return spec