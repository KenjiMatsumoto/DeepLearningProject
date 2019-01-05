
# coding: utf-8

# In[1]:


# import 
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Input, Dense, Flatten, BatchNormalization, Activation, add
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.utils import to_categorical
import argparse
import numpy as np


# In[ ]:


def rescell(data, filters, kernel_size, option=False):
    strides=(1,1)
    if option:
        strides=(2,2)
    x=Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding="same")(data)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)
    
    data=Conv2D(filters=int(x.shape[3]), kernel_size=(1,1), strides=strides, padding="same")(data)
    x=Conv2D(filters=filters,kernel_size=kernel_size,strides=(1,1),padding="same")(x)
    x=BatchNormalization()(x)
    x=add([x,data])
    x=Activation('relu')(x)
    return x

def ResNet(img_rows, img_cols, img_channels, x_train):
	input=Input(shape=(img_rows,img_cols,img_channels))
	x=Conv2D(32,(7,7), padding="same", input_shape=x_train.shape[1:],activation="relu")(input)
	x=MaxPooling2D(pool_size=(2,2))(x)

	x=rescell(x,64,(3,3))
	x=rescell(x,64,(3,3))
	x=rescell(x,64,(3,3))

	x=rescell(x,128,(3,3),True)

	x=rescell(x,128,(3,3))
	x=rescell(x,128,(3,3))
	x=rescell(x,128,(3,3))

	x=rescell(x,256,(3,3),True)

	x=rescell(x,256,(3,3))
	x=rescell(x,256,(3,3))
	x=rescell(x,256,(3,3))
	x=rescell(x,256,(3,3))
	x=rescell(x,256,(3,3))

	x=rescell(x,512,(3,3),True)

	x=rescell(x,512,(3,3))
	x=rescell(x,512,(3,3))

	x=AveragePooling2D(pool_size=(int(x.shape[1]),int(x.shape[2])),strides=(2,2))(x)

	x=Flatten()(x)
	x=Dense(units=10,kernel_initializer="he_normal",activation="softmax")(x)
	model=Model(inputs=input,outputs=[x])
	return model

def train():


    # cifar10のデータ取得
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train1, x_valid, y_train1, y_valid = train_test_split(x_train, y_train, test_size=0.2)

    model = ResNet(32,32,3,x_train)
    # RGB画像で32×32なので32×32×3にreshapeする
    x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
    x_valid = x_valid.reshape(x_valid.shape[0], 32, 32, 3)
    x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
    
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
                  optimizer=Adam(),  metrics=['accuracy'])
    # 学習
    model.fit(x_train, y_train, epochs=40, batch_size=128, verbose=1, validation_data=(x_valid, y_valid))
    # 精度算出
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


# In[3]:


train()

