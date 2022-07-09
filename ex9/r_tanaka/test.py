import IPython
import librosa
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score  # モデル評価用(正答率)
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')


def display(*dfs, head=True):
    """データフレームを整えて出力

    Args:
        head (bool, optional): _description_. Defaults to True.
    """
    for df in dfs:
        IPython.display.display(df.head() if head else df)


def feature_extraction(path_list):
    """
    wavファイルのリストから特徴抽出を行い, リストで返す
    扱う特徴量はMFCC13次元の平均（0次は含めない）
    Args:
        path_list: 特徴抽出するファイルのパスリスト
    Returns:
        features: 特徴量
    """

    load_data = (lambda path: librosa.load(path)[0])

    data = list(map(load_data, path_list))
    features = np.array([np.mean(librosa.feature.mfcc(y=y, n_mfcc=13), axis=1) for y in data])

    return features


def main():

    # テストデータ
    test = pd.read_csv("test_truth.csv")
    X_test = feature_extraction(test["path"].values)
    Y_test = np.array(test["label"].values)

    # 学習済みモデルの読み込み
    with open('model_acc:0.9259259259259259.pickle', mode='rb') as f:  # with構文でファイルパスとバイナリ読み来みモードを設定
        model = pickle.load(f)                  # オブジェクトをデシリアライズ

    # バリデーションデータの予測 (予測クラス(0~9)を返す)
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)  # 一番大きい予測確率のクラスを予測クラスに

    # 真値と予測値の表示
    df_pred = pd.DataFrame({'true value': Y_test, 'predicted value': y_pred})
    print("Display true and predicted values")
    print("---------------------------------")
    print(df_pred)
    print("---------------------------------")

    # 真値と予測確率の表示
    df_pred_prob = pd.DataFrame({'true label': Y_test,
                                 'P(0)': y_pred_prob[:, 0],
                                 'P(1)': y_pred_prob[:, 1],
                                 'P(2)': y_pred_prob[:, 2],
                                 'P(3)': y_pred_prob[:, 3],
                                 'P(4)': y_pred_prob[:, 4],
                                 'P(5)': y_pred_prob[:, 5],
                                 'P(6)': y_pred_prob[:, 6],
                                 'P(7)': y_pred_prob[:, 7],
                                 'P(8)': y_pred_prob[:, 8],
                                 'P(9)': y_pred_prob[:, 9]
                                 })

    print("Display true values and predicted probabilities")
    print("-----------------------------------------------")
    print(df_pred_prob)
    print("-----------------------------------------------")

    # モデル評価
    # acc : 正答率
    acc = accuracy_score(Y_test, y_pred) * 100
    print('Acc :', acc, '%')

    plt.figure()
    cm_fwd = confusion_matrix(y_pred, Y_test)
    sns.heatmap(cm_fwd, cmap='Blues', square=True)
    plt.title(f"Result \n (Acc. {acc:.5f}%)")
    plt.xlabel("Ground truth")
    plt.ylabel("Predicted")
    plt.savefig("confusion_matrix.png")


if __name__ == "__main__":
    main()
