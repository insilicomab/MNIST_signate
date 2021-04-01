# -*- coding: utf-8 -*-

'''
データの読み込みと確認
'''

# ライブラリのインポート
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from skimage import io
import os
from glob import glob
import zipfile
import io

from sklearn.model_selection import KFold

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint

# ランダムシードの設定
import random
np.random.seed(1234)
random.seed(1234)
tf.random.set_seed(1234)

# 入力と出力を指定
im_rows = 28 # 画像の縦ピクセルサイズ
im_cols = 28 # 画像の横ピクセルサイズ
im_color = 1 # 画像の色空間/グレイスケール
in_shape = (im_rows, im_cols, im_color)
num_classes = 10 # クラス数の定義:4クラス

# MNISTのデータを読み込み 
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# 学習データとテストデータの連結
X = np.concatenate([X_train, X_test])
y = np.concatenate([y_train, y_test])

# 画像データを三次元配列に変換
X = X.reshape(-1, im_rows, im_cols, im_color)
X = X.astype('float32') / 255

# ラベルデータをone-hotベクトルに直す
y = keras.utils.to_categorical(y, num_classes)

'''
モデリングと評価
'''

# 5分割する
folds = 5
kf = KFold(n_splits=folds)
index = 0

for train_index, val_index in kf.split(X):
    x_train = X[train_index]
    x_valid = X[val_index]
    y_train = y[train_index]
    y_valid = y[val_index]
    
    # モデルを保存するファイルパス
    filepath = './model/mnist_CNN_model[%d].h5' % index
    index += 1
    
    # CNNモデルを定義
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=in_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    
    # モデルをコンパイル
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    # 過学習の抑制
    early_stopping = EarlyStopping(monitor='val_loss', patience=10 , verbose=1)
    
    # 評価に用いるモデル重みデータの保存
    checkpointer = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
    
    # 学習を実行
    hist = model.fit(x_train, y_train,
                     batch_size=128, epochs=30,
                     verbose=1,
                     validation_data=(x_valid, y_valid),
                     callbacks=[early_stopping, checkpointer])  # CallBacksに設定
    
    # モデルを評価
    score = model.evaluate(x_valid, y_valid, verbose=1)
    print('正解率=', score[1], 'loss=', score[0])
    
    '''
    学習過程のグラフ化
    '''
    
    # 正解率の推移をプロット
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('Accuracy')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # ロスの推移をプロット
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Loss')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

# 評価に用いるモデル構造の保存
def save_model(model):
    json_string = model.to_json()
    if not os.path.isdir("cache"):
        os.mkdir("cache")
    json_name = "architecture.json"
    open(os.path.join("cache", json_name),"w").write(json_string)

save_model(model)

'''
テストデータの予測
'''

# ライブラリのインポート
from keras.models import load_model
from keras.models import model_from_json
from scipy import stats

# 入力と出力を指定
im_rows = 28 # 画像の縦ピクセルサイズ
im_cols = 28 # 画像の横ピクセルサイズ
im_color = 1 # 画像の色空間/グレイスケール
in_shape = (im_rows, im_cols, im_color)
num_classes = 10 # クラス数の定義:4クラス

# ラベルデータの読み込み
test_labels = pd.read_csv('./data/sample_submit.tsv', sep='\t', header=None)
print(test_labels.head())

# test_labelsのリスト型（list）をまとめる
test_dic = dict(zip(test_labels[0], test_labels[1])) # id: keys, class_num: value

# zipファイルのパス
zip_path = './data/test.zip'

# 画像データの格納リスト
X_test = [] # 画像のピクセル値とラベルを格納するリストを生成（説明変数）

# zipの読み込み
with zipfile.ZipFile(zip_path, 'r') as zip_file:
    for i in test_dic.keys():
        with zip_file.open('test/'+i) as img_file:
            # 画像のバイナリデータを読み込む
            img_bin = io.BytesIO(img_file.read())
            # バイナリデータをpillowから開く
            img = Image.open(img_bin)
            # グレースケールに変換
            img = img.convert('L')
            # 画像データを配列化
            img_array = np.array(img)
            # 格納リストに追加
            X_test.append(img_array)

# 画像の確認
for i in range(10):
    plt.imshow(X_test[i], cmap='gray')
    plt.show()

# np.arrayに変換
X_test = np.array(X_test)

# 読み込んだデータをの三次元配列に変換
X_test = X_test.reshape(-1, im_rows, im_cols, im_color)
X_test = X_test.astype('float32') / 255

# 予測データの格納リスト
preds = []

# 保存したモデル重みデータとモデル構造の読み込み
for index in range(0,5):
    filepath = './model/mnist_CNN_model[%d].h5' % index
    json_name = 'architecture.json'
    model = model_from_json(open(os.path.join("cache", json_name)).read())
    model.load_weights(filepath)

    # 各モデルにおける推測確率の計算　
    pred = model.predict(X_test)
    pred_max = np.argmax(pred, axis=1)
    preds.append(pred_max)

# アンサンブル学習
preds_array = np.array(preds)
pred = stats.mode(preds_array)[0].T # 予測データリストのうち最頻値を算出し、行と列を入れ替え

'''
提出
'''

# 提出用データの読み込み
sub = pd.read_csv('./data/sample_submit.tsv', sep='\t', header=None)
print(sub.head())

# 目的変数カラムの置き換え
sub[1] = pred

# ファイルのエクスポート
sub.to_csv('./submit/mnist_keras_CNN_leak_baseline.tsv', sep='\t', header=None, index=None)