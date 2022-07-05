import pickle
import os


from tensorflow.python.keras import models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
from keras.utils import np_utils

DATA_LENGTH = 4096


def load_wav_test():
    """load test wave file

    Returns:
        ndarray, axis=(sample, time): wav
        ndarray, axis=(sample): answer label
    """
    if os.path.isfile('wav_test.pickle') and os.path.isfile('ans_test.pickle'):
        with open('wav_test.pickle', 'rb') as f:
            wav = pickle.load(f)
        with open('ans_test.pickle', 'rb') as f:
            label = pickle.load(f)
    else:
        test_csv = pd.read_csv("../test.csv", dtype=str, encoding='utf8')
        wav = np.zeros((len(test_csv), DATA_LENGTH))
        label = np.zeros(len(test_csv), dtype=np.int16)
        for i, row in test_csv.iterrows():
            wav_tmp, _ = librosa.load(f"../{row.path}", sr=SAMPLING_RATE, mono=True)
            # zero padding at end of wav
            if (len(wav_tmp) > DATA_LENGTH):
                wav[i] = wav_tmp[0:DATA_LENGTH]
            else:
                wav[i, 0:len(wav_tmp)] = wav_tmp
            label[i] = row.label
        with open('wav_test.pickle', 'wb') as f:
            pickle.dump(wav, f)
        with open('ans_test.pickle', "wb") as f:
            pickle.dump(label, f)
    return wav, label


def main():
    # read data
    x, answer_label = load_wav_train()

    # convert to mfcc
    tmp_test = []
    for i in range(len(x)):
        tmp_test.append(librosa.feature.mfcc(x[i]))
    x = np.array(tmp_test)

    # load model
    model_arc_filename="./keras_model/2022-07-05 23:25:48.287827model_architecture.json"
    model_weight_filename="./keras_model/2022-07-05 23:25:48.287827model_weight.hdf5"
    model_arc_str = open(model_arc_filename).read()
    model = models.model_from_json(model_arc_str)
    model.load_weights(model_weight_filename)

    # predict
    predict = model.predict(x, verbose=0)
    predict = np.argmax(predict, axis=1)  # decode one-hot

    # accuracy
    collect_count = np.count_nonzero(answer_label == predict)
    all_count = len(answer_label)
    print(f"accuracy: {(collect_count / answer_label)}")
    print(f"        ( {collect_count} / {all_count} )")


if __name__ == "__main__":
    main()
