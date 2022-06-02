import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from my_functions import mfcc


def main():
    parser = argparse.ArgumentParser(description='ex5')
    parser.add_argument("-i", "--input", help="input file name", type=int)
    args = parser.parse_args()
    pass


if __name__ == "__main__":
    main()
