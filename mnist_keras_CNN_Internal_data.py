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