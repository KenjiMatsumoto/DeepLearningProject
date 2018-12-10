
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import chainer
import chutil

import chainer.functions as F
import chainer.links as L
from chainer import Chain
import chainer.optimizers as optimizers

from chainer.datasets.mnist import get_mnist
from chainer import optimizers, training
from chainer.training import extensions

# データセットがダウンロード済みでなければ、ダウンロードも行う
train, test = get_mnist(withlabel=True, ndim=1)
train, validation = chainer.datasets.split_dataset_random(train, 50000, seed=0)


# In[20]:


class MyConvNet(Chain):
    def __init__(self):
        super(MyConvNet, self).__init__()
        with self.init_scope():
            # 畳み込み層の定義
            # in_channels:Noneを指定しても動的にメモリ確保するので問題なく動作する
            # out_channels:出力する配列のチャンネル数
            # ksize:フィルタのサイズ（平行移動するフィルターの長さを指定）
            # stride:入力データに対してstride分フィルターを適用していくパラメータを指定
            # pad:イメージは画像データの周りにpadのサイズ分だけ空白を用意してそこに対してもフィルターを適用するようなイメージ
            # dilate:今回の実装では設定していないが、飛び飛びにフィルターを適用するパラメータ
            self.conv1 = L.Convolution2D(
                in_channels=None, out_channels=32, ksize=3, stride=1, pad=1)
            # 畳み込み層の定義２層目
            self.conv2 = L.Convolution2D(
                in_channels=None, out_channels=64, ksize=3, stride=1, pad=1)
            # 畳み込み層の定義３層目
            self.conv3 = L.Convolution2D(
                in_channels=None, out_channels=128, ksize=3, stride=1, pad=1)
            self.conv4 = L.Convolution2D(
                in_channels=None, out_channels=128, ksize=3, stride=1, pad=1)
            self.fc5 = L.Linear(None, 1000)
            self.fc6 = L.Linear(None, 10)

    def __call__(self, x):
        h = F.sigmoid(self.conv1(x.reshape((-1, 1, 28, 28))))
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = F.sigmoid(self.conv2(h))
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = F.sigmoid(self.conv3(h))
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = F.sigmoid(self.conv4(h))
        h = F.sigmoid(self.fc5(h))
        return self.fc6(h)
    
    def forward2(self, x):
        h = F.relu(self.conv1(x.reshape((-1, 1, 28, 28))))
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = F.relu(self.conv2(h))  # sigmoid -> relu
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = F.relu(self.conv3(h))  # sigmoid -> relu
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = F.relu(self.conv4(h))  # sigmoid -> relu
        h = F.relu(self.fc5(h))  # sigmoid -> relu
        return self.fc6(h)


# In[21]:


def  train_and_validate(
        model, optimizer, train, validation, n_epoch, batchsize, device=0):
    
    # 1. deviceがgpuであれば、gpuにモデルのデータを転送する
    if device >= 0:
        model.to_cpu()
        
    # 2. Optimizerを設定する
    optimizer.setup(model)
    
    # 3. DatasetからIteratorを作成する
    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    validation_iter = chainer.iterators.SerialIterator(
        validation, batchsize, repeat=False, shuffle=False)
    
    # 4. Updater・Trainerを作成する
    updater = training.StandardUpdater(train_iter, optimizer)
    trainer = chainer.training.Trainer(updater, (n_epoch, 'epoch'), out='out')
    
    # 5. Trainerの機能を拡張する
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.Evaluator(validation_iter, model), name='val')
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'main/accuracy', 'val/main/loss', 'val/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.PlotReport(
        ['main/loss', 'val/main/loss'],x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(
        ['main/accuracy', 'val/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
    trainer.extend(extensions.dump_graph('main/loss'))
    
    # 6. 訓練を開始する
    trainer.run()


# In[ ]:


n_epoch = 20
batchsize = 128

model = MyConvNet()
classifier_model = L.Classifier(model)
optimizer = optimizers.Adam()
train_and_validate(
    classifier_model, optimizer, train, validation, n_epoch, batchsize)

