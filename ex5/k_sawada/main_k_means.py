import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from my_functions import k_means


def main():
    parser = argparse.ArgumentParser(description='ex5')
    parser.add_argument("-i", "--input", help="input file id", type=int)
    parser.add_argument("-c", "--classes", help="classes count", type=int)
    args = parser.parse_args()
    
    # read data
    data = pd.read_csv(f"../data{args.input}.csv").values

    classified = k_means.k_means(data, args.classes)


if __name__ == "__main__":
    main()
