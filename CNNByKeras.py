
# coding: utf-8

# In[6]:


# import 
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


# In[7]:


# mnistデータの取得
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train1, x_vaild, y_train1, y_valid = train_test_split(x_train, y_train, test_size=0.175)


# In[9]:


# model作成 CNNByChainerと同じ層構成にする
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='sigmoid', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='sigmoid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='sigmoid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='sigmoid'))
# model.add(Dropout(0.25))
# 入力を平滑化する層（いわゆるデータをフラット化する層、例えば4次元配列を1次元配列に変換するなど）
model.add(Flatten())
# denseは全結合層
model.add(Dense(1000, activation='sigmoid'))
# dropout
# model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

