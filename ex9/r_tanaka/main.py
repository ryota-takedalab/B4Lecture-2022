import lightgbm as lgb
import librosa
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split  # データセット分割用
from sklearn.metrics import accuracy_score  # モデル評価用(正答率)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


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
    # データの読み込み
    training = pd.read_csv("training.csv")
    # 学習データの特徴抽出
    X_train = feature_extraction(training["path"].values)
    Y_train = np.array(training["label"].values)

    # 学習データとバリデーションデータに分割 (バリデーションセット20%)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train, Y_train,
        test_size=0.20,
        random_state=2,
    )

    # lightGBM
    # 学習に使用するデータを設定
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_validation, y_validation, reference=lgb_train)

    # LightGBM parameters
    params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'multiclass',  # 目的: 多クラス分類
            'num_class': 10,  # クラス数: 10
            'metric': {'multi_logloss'},  # 評価指標: Multi-class logloss
            'learning_rate': 0.255,
            'num_leaves': 23,
            'min_data_in_leaf': 1,
            'num_iteration': 1000,  # 1000回学習
            'verbose': 0
    }

    # モデルの学習
    evals_result = {}
    model = lgb.train(params,  # パラメータ
                      lgb_train,  # トレーニングデータの指定
                      valid_sets=[lgb_train, lgb_eval],  # 検証データの指定
                      valid_names=['train', 'valid'],
                      early_stopping_rounds=20,  # 20回ごとに検証精度の改善を検討 -> 精度が改善しなければ学習を終了
                      evals_result=evals_result
                      )

    # バリデーションデータの予測 (予測クラス(0~9)を返す)
    y_pred_prob = model.predict(X_validation)
    y_pred = np.argmax(y_pred_prob, axis=1)  # 一番大きい予測確率のクラスを予測クラスに

    # 真値と予測値の表示
    df_pred = pd.DataFrame({'true value': y_validation, 'predicted value': y_pred})

    print("Display true and predicted labels")
    print("---------------------------------")
    print(df_pred)

    # 真値と予測確率の表示
    df_pred_prob = pd.DataFrame({'true label': y_validation,
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

    print("Display true labels and predicted probabilities")
    print("-----------------------------------------------")
    print(df_pred_prob)

    # モデル評価
    # acc : 正答率
    acc = accuracy_score(y_validation, y_pred)
    print('Acc :', acc)

    # 学習済みモデルの保存
    filename = 'model_acc:' + str(acc) + '.pickle'
    with open(filename, mode='wb') as f:  # with構文でファイルパスとバイナリ書き込みモードを設定
        pickle.dump(model, f)  # オブジェクトをシリアライズ

    # 学習過程の可視化
    plt.plot(evals_result['train']['multi_logloss'], label='train')
    plt.plot(evals_result['valid']['multi_logloss'], label='valid')
    plt.ylabel('Log loss')
    plt.xlabel('Boosting round')
    plt.title(f'Training performance \n Acc: {acc}')
    plt.legend()
    plt.show()
    plt.savefig("train_history.png")


if __name__ == "__main__":
    main()
