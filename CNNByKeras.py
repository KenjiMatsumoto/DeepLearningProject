
# coding: utf-8

# In[1]:


# import 
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Input, Dense, Flatten
from keras.optimizers import SGD
from keras.losses import categorical_crossentropy
from keras.utils import to_categorical
import argparse
import numpy as np


# In[3]:


# model作成 CNNByChainerと同じ層構成にする
def create_CNN_model(input_shape=(28, 28, 1), class_num=10):
    input = Input((28, 28, 1))
    kernel_size = (3, 3)
    max_pool_size = (2, 2)
    # 畳み込み層の実装
    # 1層目
    cnn = Conv2D(32, kernel_size=kernel_size, padding='same', strides=(1, 1), activation='sigmoid', input_shape=(28, 28, 1))(input)
    cnn = MaxPooling2D(kernel_size=(2,2), pool_size=max_pool_size, strides=(2, 2))(cnn)
    cnn = Conv2D(64, kernel_size, padding='same', strides=(1, 1), activation='sigmoid')(cnn)
    cnn = MaxPooling2D(kernel_size=(2,2), pool_size=max_pool_size, strides=(2, 2))(cnn)
    cnn = Conv2D(128, kernel_size, padding='same', strides=(1, 1), activation='sigmoid')(cnn)
    cnn = MaxPooling2D(kernel_size=(2,2), pool_size=max_pool_size, strides=(2, 2))(cnn)
    cnn = Conv2D(128, kernel_size, padding='same', strides=(1, 1), activation='sigmoid')(cnn)
    # 入力を平滑化する層（いわゆるデータをフラット化する層、例えば4次元配列を1次元配列に変換するなど）
    fc = Flatten()(cnn)
    # denseは全結合層
    fc = Dense(1000, activation='sigmoid')(fc)
    softmax = Dense(10, activation='softmax')(fc)
    model = Model(input=input, output=softmax)
    
    return model

# 全体の正解率を算出
def predict_accuracy(x_batch, y_batch, model):
    # 予測確率
    preds = model.predict(x_batch, verbose=0)
    batch_size = len(x_batch)
    count = 0
    # y(正解の値)とpred(予測の値)を比較
    for (y, pred) in zip(y_batch, preds):
        # 正解かどうかをチェックし、正解した場合カウントする
        if np.argmax(y) == np.argmax(pred):
            count += 1
    print('acc ', (count/batch_size))

def train():

    model = create_CNN_model()
    # mnistデータの取得
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train1, x_valid, y_train1, y_valid = train_test_split(x_train, y_train, test_size=0.175)
    # グレースケールの画像で28×28なので28×28×1にreshapeする
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_valid = x_valid.reshape(x_valid.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    x_train = x_train.astype('float32')
    x_valid = x_valid.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_valid /= 255
    x_test /= 255

    # one-hot vector形式に変換する
    y_train = to_categorical(y_train, 10)
    y_valid = to_categorical(y_valid, 10)
    y_test = to_categorical(y_test, 10)
    
    model.compile(loss=categorical_crossentropy,
                  optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True),  metrics=['accuracy'])
    # 学習
    model.fit(x_train, y_train, epochs=20, batch_size=128, verbose=1, validation_data=(x_valid, y_valid))
    # 精度算出
#     score = model.evaluate(x_test, y_test, verbose=0)
#     print('Test loss:', score[0])
#     print('Test accuracy:', score[1])
    predict_accuracy(x_test, y_test, model)


# In[ ]:


train()

