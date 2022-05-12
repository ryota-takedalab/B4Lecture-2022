import argparse

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import librosa
import librosa.display
import scipy.io.wavfile


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ex2 LPF')
    parser.add_argument("-i", "--input", help="input file")
    args = parser.parse_args()
    