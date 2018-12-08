
# coding: utf-8

# ## ニューラルネットワークの構築

# In[1]:


# 必要なライブラリのインポート
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定

from common import functions
from common import optimizer
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_mldata
# データセットのロード
mnist = fetch_mldata('MNIST original', data_home=".")
print(mnist.data.shape)
print(mnist.target)
x_vals = mnist.data
y_vals = mnist.target
Y_vals = np.eye(10)[y_vals.astype(int)] # 1-of-K表現に変換
# トレーニングデータ（60％）とテストデータ（40％）に分割
x_train, x_test, y_train, y_test = train_test_split(x_vals, Y_vals, test_size=0.4, shuffle=True)


# In[10]:


class LayerNet:
    
    # コンストラクタ
    def __init__(self):
        self.input_size = 784
        self.hidden_size1 = 300
        self.hidden_size2 = 150
        self.output_size = 10
        self.batch_size = 80
        self.weight_init = 0.001
        self.learning_rate = 0.005
    
    # ネットワークの初期化を実施
    def init_network(self):
        network = {}
    
        # 重みの設定
        # 通常設定
        network['W1'] = self.weight_init * np.random.randn(self.input_size, self.hidden_size1)
        network['W2'] = self.weight_init * np.random.randn(self.hidden_size1, self.hidden_size2)
        network['W3'] = self.weight_init * np.random.randn(self.hidden_size2, self.output_size)

#         # Xavierでの設定
#         network['W1'] = np.random.randn(input_layer_size, hidden_layer_size1) / (np.sqrt(input_layer_size))
#         network['W2'] = np.random.randn(hidden_layer_size1, hidden_layer_size2) / (np.sqrt(hidden_layer_size1))
#         network['W3'] = np.random.randn(hidden_layer_size2, output_layer_size) / (np.sqrt(hidden_layer_size2))

        # Heでの設定
#         network['W1'] = np.random.randn(self.input_size, self.hidden_size1) / (np.sqrt(self.input_size)) * np.sqrt(2)
#         network['W2'] = np.random.randn(self.hidden_size1, self.hidden_size2) / (np.sqrt(self.hidden_size1)) * np.sqrt(2)
#         network['W3'] = np.random.randn(self.hidden_size2, self.output_size) / (np.sqrt(self.hidden_size2)) * np.sqrt(2)

        # バイアスの設定
        network['b1'] = np.zeros(self.hidden_size1)
        network['b2'] = np.zeros(self.hidden_size2)
        network['b3'] = np.zeros(self.output_size)

        return network
    
    def gradient(self, network, x_vec, y_vec):
        # ランダムにバッチを取得    
        batch_mask = np.random.choice(len(x_vec), self.batch_size)
        # ミニバッチに対応する教師訓練ラベルデータを取得    
        x_batch = x_vec[batch_mask]
        # ミニバッチに対応する訓練正解ラベルデータを取得する
        y_batch = y_vec[batch_mask]
#         y_batch = y_batch[:, np.newaxis]

        z1, z2, y = self.forward(network, x_batch)
        grad = self.backward(x_batch, y_batch, z1, z2, y)

        # optimizerの設定 モメンタムを利用
        opt = optimizer.Adam(self.learning_rate)
        opt.update(network, grad)
        
        return y_batch, y
    
    # 順伝播
    def forward(self, network, x):
        W1, W2, W3 = network['W1'], network['W2'], network['W3']
        b1, b2, b3 = network['b1'], network['b2'], network['b3']

        # 勾配
        u1 = np.dot(x, W1) + b1
        # 活性化関数 Relu関数を使用
        z1 = functions.sigmoid(u1)
        # 勾配
        u2 = np.dot(z1, W2) + b2
        # 活性化関数 Relu関数を使用
        z2 = functions.sigmoid(u2)
        # 勾配
        u3 = np.dot(z2, W3) + b3
        # 誤差関数(ソフトマックス関数)
        y = functions.softmax(u3)

        return z1, z2, y

    # 逆伝播
    def backward(self, x, d, z1, z2, y):
        grad = {}

        W1, W2, W3 = network['W1'], network['W2'], network['W3']
        b1, b2, b3 = network['b1'], network['b2'], network['b3']

        # 出力層でのデルタ 
        delta3 = functions.d_softmax_with_loss(d, y)
        # b3の勾配
        grad['b3'] = np.sum(delta3, axis=0)
        # W3の勾配
        grad['W3'] = np.dot(z2.T, delta3)
        # 活性化関数の導関数 Relu関数
        delta2 = np.dot(delta3, W3.T) * functions.d_sigmoid(z2)
        # b2の勾配
        grad['b2'] = np.sum(delta2, axis=0)
        # W2の勾配
        grad['W2'] = np.dot(z1.T, delta2)
        # 活性化関数の導関数 Relu関数
        delta1 = np.dot(delta2, W2.T) * functions.d_sigmoid(z1)
        # b1の勾配
        grad['b1'] = np.sum(delta1, axis=0)
        # W1の勾配
        grad['W1'] = np.dot(x.T, delta1)

        return grad


# In[11]:


# 学習回数(1000回)
learning_num = 2000

# 描写頻度
plot_interval=10

layerNet = LayerNet()

# パラメータの初期化
network = layerNet.init_network()

losses = []
losses_test = []

for i in range(learning_num):
    # 訓練用の学習
    y_batch, y = layerNet.gradient(network, x_train, y_train)

    # テスト用の学習
    y_test_batch, y_t = layerNet.gradient(network, x_test, y_test)
    
    if (i + 1) % plot_interval == 0:
        loss = functions.least_square(y_batch, y)
        losses.append(loss)
        print('Generation: ' + str(i+1) + '. 誤差 = ' + str(loss))
        loss_test = functions.least_square(y_test_batch, y_t)
        losses_test.append(loss_test)
        print('Generation_Test: ' + str(i+1) + '. 誤差(テスト) = ' + str(loss_test))

lists = range(0, learning_num, plot_interval)
plt.plot(lists, losses, label="training set")
plt.plot(lists, losses_test, label="test set")
plt.legend(loc="lower right")
plt.title("loss")
plt.xlabel("count")
plt.ylabel("loss")
# グラフの表示
plt.show()

