import pickle
import os

import tensorflow as tf
from tensorflow import keras as kr
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import callbacks
from tensorflow.python.keras import backend as K

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
from scipy import signal
from sklearn.model_selection import train_test_split

SAMPLING_RATE = 8000  # by wave format data
LABELS = 10  # number of labels to classify
HIDDEN_UNITS = 128
# https://qiita.com/everylittle/items/ba821e93d275a421ca2b
MAX_LENGTH = 18262
SPLIT_SEED = 0


def get_max_length():
    """show max wav length in dataset
    """
    train_csv = pd.read_csv("../training.csv", dtype=str, encoding='utf8')
    test_csv = pd.read_csv("../test.csv", dtype=str, encoding='utf8')
    train_length = len(train_csv)
    count = train_length + len(test_csv)
    length = np.zeros(count)
    for i, row in train_csv.iterrows():
        length[i] = len(librosa.load(f"../{row.path}", sr=SAMPLING_RATE, mono=True)[0])
    for i, row in test_csv.iterrows():
        length[i + train_length] = len(librosa.load(f"../{row.path}", sr=SAMPLING_RATE, mono=True)[0])
    print(f"max_length: {np.max(length)} ({np.argmax(length)})")  # > 3000
    
    # path_list = sorted(glob.glob("../free-spoken-digit-dataset/recordings/*"))
    # count = len(path_list)
    # length = np.zeros(count)
    # for i in range(len(path_list)):
    #     length[i] = len(librosa.load(path_list[i], sr=SAMPLING_RATE, mono=True)[0])
    # print(f"max_length: {np.max(length)}")  # > 3000


def preview_data():
    train_csv = pd.read_csv("../training.csv", dtype=str, encoding='utf8')    
    print(train_csv)
    row = train_csv.iloc[0]
    print(row)
    wav, _ = librosa.load(f"../{row.path}", sr=SAMPLING_RATE, mono=True)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(wav)
    plt.show()


def load_wav_train():
    """load train wave file

    Returns:
        ndarray, axis=(sample, time): wav
        ndarray, axis=(sample): answer label
    """
    train_csv = pd.read_csv("../training.csv", dtype=str, encoding='utf8')
    wav = np.zeros((len(train_csv), MAX_LENGTH))
    label = np.zeros(len(train_csv))
    for i, row in train_csv.iterrows():
        wav_tmp, _ = librosa.load(f"../{row.path}", sr=SAMPLING_RATE, mono=True)
        # zero padding at end of wav
        wav[i, 0:len(wav_tmp)] = wav_tmp
        label[i] = row.label
    with open('wav.pickle', 'wb') as f:
        pickle.dump(wav, f)
    with open('ans.pickle', "wb") as f:
        pickle.dump(label, f)
    return wav, label


def wav2spectrogram(wav):
    _, _, specs = signal.spectrogram(wav)
    return specs


def main():
    if os.path.isfile('wav.pickle') and os.path.isfile('ans.pickle'):
        with open('wav.pickle', 'rb') as f:
            x = pickle.load(f)
        with open('ans.pickle', 'rb') as f:
            answer_label = pickle.load(f)
    else:
        x, answer_label = load_wav_train()
    
    x_train, x_test = train_test_split(x, random_state=SPLIT_SEED)
    ans_train, ans_test = train_test_split(answer_label, random_state=SPLIT_SEED)
    
    x_train = wav2spectrogram(x_train)
    print(x_train.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(x_train[0])
    plt.show()
    
    # dummy sample
    # x_train = np.zeros((100, 2500, 1800,))  # (sample, time, frequency)
    input_shape = x_train.shape[1:]
    model = models.Sequential()
    # https://www.tensorflow.org/guide/keras/masking_and_padding?hl=ja
    model.add(layers.Masking(input_shape=input_shape, mask_value=-1.0))
    model.add(layers.LSTM(HIDDEN_UNITS))
    model.add(layers.Dense(LABELS))
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.summary()
    
    fit_callbacks = [
        callbacks.EarlyStopping(monitor='val_loss',
                                patience=5,
                                mode='min')
    ]

    # モデルを学習させる
    model.fit(x_train, ans_train,
              epochs=1000,
              batch_size=4096,
              shuffle=True,
              validation_data=(x_test, ans_test),
              callbacks=fit_callbacks,
              )

    # テストデータの損失を確認しておく
    score = model.evaluate(x_test, ans_test, verbose=0)
    print('test xentropy:', score)


if __name__ == '__main__':
    # get_max_length()
    # preview_data()
    main()
