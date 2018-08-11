import numpy as np

# 確率的勾配降下法
class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        
    def update(self, params, grad):
        for key in params.keys():
            # パラメータを更新する為に、現在のパラメータ値から(学習率 * 誤差逆伝播から導いた値)をマイナス
            params[key] -= self.learning_rate * grad[key] 

# モメンタム
class Momentum:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grad):
        if self.v is None:
            self.v = {}
            for key, val in params.items(): 
                # W or bの各配列の初期化
                self.v[key] = np.zeros_like(val)
                
        for key in params.keys():
            # (モメンタム * 一つ前に算出 V) - (学習率 * 誤差逆伝播から導いた値)
            self.v[key] = self.momentum * self.v[key] - self.learning_rate * grad[key] 
            # 算出した V をパラメータに加算
            params[key] += self.v[key]

# NAG
class Nesterov:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grad):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                # W or bの各配列の初期化
                self.v[key] = np.zeros_like(val)
            
        for key in params.keys():
            # V にモメンタム値を乗算
            self.v[key] *= self.momentum
            # (学習率 * 誤差逆伝播から導いた値)を差し引く
            self.v[key] -= self.learning_rate * grad[key]
            # モメンタム値 * モメンタム値 * V をパラメータに加算
            params[key] += self.momentum * self.momentum * self.v[key]
            # (1 + モメンタム値) * 学習率 * 誤差逆伝播から導いた値のトータルをパラメータから差し引く
            params[key] -= (1 + self.momentum) * self.learning_rate * grad[key]

# AdaGrad
class AdaGrad:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.h = None
        
    def update(self, params, grad):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                # W or bの各配列の初期化
                self.h[key] = np.zeros_like(val)
            
        for key in params.keys():
            # 誤差逆伝播から導いた値の２乗
            self.h[key] += grad[key] * grad[key]
            # 学習率 * 誤差逆伝播から導いた値 / (h の平方根 + ごく微小な値 θ)
            params[key] -= self.learning_rate * grad[key] / (np.sqrt(self.h[key]) + 1e-7)

# RMSprop
class RMSprop:
    def __init__(self, learning_rate=0.01, decay_rate = 0.99):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.h = None
        
    def update(self, params, grad):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                # W or bの各配列の初期化
                self.h[key] = np.zeros_like(val)
            
        for key in params.keys():
            # hにα↓だとdecay_rateを乗算
            self.h[key] *= self.decay_rate
            # (1 - α) * 勾配(誤差逆伝播から導いた値)の２乗 
            self.h[key] += (1 - self.decay_rate) * grad[key] * grad[key]
            # 最後の計算はAdagradと同様
            params[key] -= self.learning_rate * grad[key] / (np.sqrt(self.h[key]) + 1e-7)

# Adam
class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grad):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1
        lr_t  = self.learning_rate * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)         
        
        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grad[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grad[key] ** 2 - self.v[key])
            
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
