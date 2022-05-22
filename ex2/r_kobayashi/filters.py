import numpy as np
from math import pi

def sinc(x):
    """ 
    return sinc function.

    Parameters:
        x : np.ndarray
            input array

    Return:
        np.ndarray
            array applied sinc function.
            sinc(x) = 1             x = 0
                    = sin(x) / x    othewise
    """

    return np.where(x ==0, 1, np.sin(x) / x)


def LPF(sr, fc, fir_size = 255):
    """ 
    Low Pass Filter(LPF)
    
    Parameters:
        sr : int
            sampling rate
        fc : int
            cutoff frequency
        fir_size : int, default 255
            filter size
            the number of filter coefficients
        
        Returns:
            np.ndarray
                Low Pass Filter
    """

    # cutoff angular frequency
    wc = 2 * pi * fc
    # normalized angular frequency
    wcn = wc / sr
    
    # create filter
    arange = np.arange(-fir_size // 2, fir_size // 2 + 1)
    fir = wcn * sinc(wcn * arange) * pi
    #apply window function method
    window = np.hamming(fir_size + 1)

    return fir * window

def HPF(sr, fc, fir_size=255):
    """
    High Pass Filter(HPF)
    Parameters:
        sr : int
            sampling rate
        fc : int
            cutoff frequency
        fir_size : int, default 255
            filter size
            the number of filter coefficients
    Returns:
        np.ndarray
            High Pass Filter
    """

    # cutoff angular frequency
    wc = 2 * pi * fc
    # normalized angular freqency
    wcn = wc / sr

    # create filter
    arange = np.arange(-fir_size // 2, fir_size // 2 + 1)
    fir = sinc(pi * arange) - wcn * sinc(wcn * arange) / pi

    # apply window function method
    window = np.hamming(fir_size + 1)

    return fir * window


def BPF(sr, f_low, f_high, fir_size=255):
    """
    Band Pass Filter(BPF)
    Parameters:
        sr : int
            sampling rate
        f_low : int
            low edge frequency
        f_high : int
            high edge frequency
        fir_size : int, default 255
            filter size
            the number of filter coefficients
    Returns:
        np.ndarray
            Band Pass Filter
    """

    if f_high < f_low:
        f_high, f_low = f_low, f_high

    # cutoff angular frequency
    w_low = 2 * pi * f_low
    w_high = 2 * pi * f_high
    # normalized angular freqency
    wn_low = w_low / sr
    wn_high = w_high / sr

    # create filter
    arange = np.arange(-fir_size // 2, fir_size // 2 + 1)
    fir = (wn_high * sinc(wn_high * arange) - wn_low * sinc(wn_low * arange)) / pi

    # apply window function method
    window = np.hamming(fir_size + 1)

    return fir * window


def BEF(sr, f_low, f_high, fir_size=255):
    """
    Band Elimination Filter(BEF)
    Parameters:
        sr : int
            sampling rate
        f_low : int
            low edge frequency
        f_high : int
            high edge frequency
        fir_size : int, default 255
            filter size
            the number of filter coefficients
    Returns:
        np.ndarray
            Band Elimination Filter
    """

    if f_high < f_low:
        f_high, f_low = f_low, f_high

    # cutoff angular frequency
    w_low = 2 * pi * f_low
    w_high = 2 * pi * f_high
    # normalized angular freqency
    wn_low = w_low / sr
    wn_high = w_high / sr

    # create filter
    arange = np.arange(-fir_size // 2, fir_size // 2 + 1)
    fir = (
        sinc(pi * arange)
        + (wn_low * sinc(wn_low * arange) - wn_high * sinc(wn_high * arange)) / pi
    )

    # apply window function method
    window = np.hamming(fir_size + 1)

    return fir * window