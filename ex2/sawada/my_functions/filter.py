import numpy as np


def _create_base_filter(cutoff_frequency, nperseg=512, sr=16000):
    """filter creation base function
    
    This function returns the basic LPF for creating filter, wrapped and used.

    Args:
        cutoff_frequency (int): cutoff frequency
        nperseg (int, optional): samples in stft segment. Defaults to 512.
        sr (int, optional): sampling rate. Defaults to 16000.
        
    Returns:
        ndarray: filter
    """
    
    regularized_cutoff_frequency = 2 * np.pi * cutoff_frequency / sr
    t = np.arange(nperseg)
    
    # create filter
    filter = 2 * regularized_cutoff_frequency \
        * np.sinc(regularized_cutoff_frequency / np.pi * (t - nperseg // 2)) \
        * np.hamming(nperseg)
    return filter


def create_lpf(cutoff_frequency, fft_size=512, sr=16000):
    """low-pass filter

    Args:
        cutoff_frequency (int): cutoff frequency
        fft_size (int, optional): samples in stft segment. Defaults to 512.
        sr (int, optional): sampling rate. Defaults to 16000.

    Returns:
        ndarray: low-pass filter
    """
    return _create_base_filter(cutoff_frequency, fft_size, sr)


def create_hpf(cutoff_frequency, fft_size=512, sr=16000):
    """high-pass filter

    Args:
        cutoff_frequency (int): cutoff frequency
        fft_size (int, optional): samples in stft segment. Defaults to 512.
        sr (int, optional): sampling rate. Defaults to 16000.

    Returns:
        ndarray: high-pass filter
    """
    return _create_base_filter(sr // 2, fft_size, sr) \
        - _create_base_filter(cutoff_frequency, fft_size, sr)


def apply_filter(audio, filter):
    """apply filter to audio

    Args:
        audio (ndarray, axis=(time,)): audio array
        filter (ndarray, axis=(time,)): filter

    Returns:
        ndarray: filtered audio
    """
    # define variables often used
    filter_length = len(filter)
    filter_fliped = filter[::-1]
    
    # initialize return value
    result = np.zeros(len(audio) + filter_length - 1)
    
    audio_extended = np.concatenate([
        np.zeros(filter_length - 1),
        audio,
        np.zeros(filter_length - 1)])

    # convolve
    for i in range(len(result)):
        result[i] = np.sum(filter_fliped * audio_extended[i: i + filter_length])
    return result
