
# coding: utf-8

# In[2]:


# import
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().system('curl https://colab.chainer.org/install | sh -')
get_ipython().system('pip install chutil')
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

from chainer.datasets.cifar import get_cifar100
from chainer import optimizers, training
from chainer.training import extensions


# In[3]:


# データセットがダウンロード済みでなければ、ダウンロードも行う
train, test = get_cifar100()
train, validation = chainer.datasets.split_dataset_random(train, 40000, seed=0)


# In[ ]:


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
                in_channels=None, out_channels=256, ksize=3, stride=1, pad=1)
            self.conv5 = L.Convolution2D(
                in_channels=None, out_channels=256, ksize=3, stride=1, pad=1)
            self.fc6 = L.Linear(None, 1000)
            self.fc7 = L.Linear(None, 100)

    def __call__(self, x):
        h = F.relu(self.conv1(x.reshape((-1, 3, 32, 32))))
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h = F.relu(self.fc6(h))
        return self.fc7(h)


# In[ ]:


def  train_and_validate(
        model, optimizer, train, validation, n_epoch, batchsize, device=0):
    
    # 1. deviceがgpuであれば、gpuにモデルのデータを転送する
    if device >= 0:
        model.to_gpu(device)
        
    # 2. Optimizerを設定する
    optimizer.setup(model)
    
    # 3. DatasetからIteratorを作成する
    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    validation_iter = chainer.iterators.SerialIterator(
        validation, batchsize, repeat=False, shuffle=False)
    
    # 4. Updater・Trainerを作成する
    updater = training.StandardUpdater(train_iter, optimizer, device=device)
    trainer = chainer.training.Trainer(updater, (n_epoch, 'epoch'), out='out')
    
    # 5. Trainerの機能を拡張する
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.Evaluator(validation_iter, model, device=device), name='val')
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


# test 結果計算
def show_test_performance(model, test, batchsize, device=0):
    if device >= 0:
        model.to_gpu()
        
    test_iter = chainer.iterators.SerialIterator(
        test, batchsize, repeat=False, shuffle=False
    )
    test_evaluator = extensions.Evaluator(test_iter, model, device=device)
    results = test_evaluator()
    print("Test accuracy:", results["main/accuracy"])


# In[7]:


n_epoch = 40
batchsize = 128

model = MyConvNet()
classifier_model = L.Classifier(model)
optimizer = optimizers.Adam()
train_and_validate(
    classifier_model, optimizer, train, validation, n_epoch, batchsize)
show_test_performance(classifier_model, test, batchsize)

