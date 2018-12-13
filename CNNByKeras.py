
# coding: utf-8

# In[21]:


# import 
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Input, Dense, Flatten


# In[22]:


# mnistデータの取得
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train1, x_vaild, y_train1, y_valid = train_test_split(x_train, y_train, test_size=0.175)


# In[25]:


# model作成 CNNByChainerと同じ層構成にする
input = Input((28, 28, 1))
# 畳み込み層の実装
cnn = Conv2D(32, kernel_size=(3, 3), padding='same', activation='sigmoid', input_shape=(28, 28, 1))(input)
cnn = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(cnn)
cnn = Conv2D(64, (3, 3), padding='same', activation='sigmoid')(cnn)
cnn = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(cnn)
cnn = Conv2D(128, (3, 3), padding='same', activation='sigmoid')(cnn)
cnn = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(cnn)
cnn = Conv2D(128, (3, 3), padding='same', activation='sigmoid')(cnn)
# 入力を平滑化する層（いわゆるデータをフラット化する層、例えば4次元配列を1次元配列に変換するなど）
fc = Flatten()(cnn)
# denseは全結合層
fc = Dense(1000, activation='sigmoid')(fc)
softmax = Dense(10, activation='softmax')(fc)
model = Model(input=input, output=softmax)

