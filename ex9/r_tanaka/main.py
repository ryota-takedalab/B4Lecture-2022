import optuna.integration.lightgbm as lgb
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split  # データセット分割用
from sklearn.metrics import accuracy_score  # モデル評価用(正答率)
import matplotlib.pyplot as plt
import math
import warnings
warnings.filterwarnings('ignore')


def main():
    # 前処理済み学習データの読み込み
    X_train = np.load('X_train.npy')
    Y_train = np.load('Y_train.npy')

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
            "objective": "multiclass",
            "boosting_type": "gbdt",
            "num_class": 10,
            "metric": "multi_logloss",
            "verbosity": -1,
    }

    # モデルの学習
    # パラメータはOptuneにより自動最適化される
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

    # 真値と推定値の表示
    df_pred = pd.DataFrame({'true value': y_validation, 'predicted value': y_pred})

    print("Display true and predicted labels")
    print("---------------------------------")
    print(df_pred)

    # 真値と推定確率の表示
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
    filename = 'model/model_acc' + str(math.floor(acc * 100000)) + '.pickle'
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

    # Optuneにより最適化されたパラメータの取得
    best_params = model.params
    print("  Params: ")
    print(best_params)


if __name__ == "__main__":
    main()
