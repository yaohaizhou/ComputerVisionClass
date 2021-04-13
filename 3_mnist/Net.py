import numpy as np
from typing import List
from collections import defaultdict
import pdb


class FCLayer():
    def __init__(self, net_structure: List):
        feature_in, feature_out = net_structure
        self.W_data = np.random.randn(
            feature_in, feature_out) / np.sqrt(feature_in/2)
        self.b_data = np.random.randn(1, feature_out)
        self.W_velocity = np.zeros_like(self.W_data)
        self.b_velocity = np.zeros_like(self.b_data)

    def __call__(self, x: np.array):
        return self.forward(x)

    def forward(self, x: np.array):
        self.x_ = x
        return np.dot(x, self.W_data) + self.b_data

    def backward(self, delta: np.array):
        backward_loss = np.dot(delta, (self.W_data).T)
        self.W_grad = np.dot((self.x_).T, delta)
        self.b_grad = np.sum(delta, axis=0, keepdims=True)
        return backward_loss


class Conv():
    def __init__(self, kernel_size: int, padding: int, stride: int, in_features: int, out_features: int):
        self.k = kernel_size
        self.in_features = in_features
        self.out_features = out_features
        self.padding = padding
        self.stride = stride
        # self.W_data = np.ones([self.in_features, self.k, self.k, self.out_features])
        self.W_data = np.random.randn(self.in_features, self.k, self.k, self.out_features)
        # self.b_data = np.zeros([1, self.out_features])
        self.b_data = np.random.randn(1, self.out_features)

    def __call__(self, x: np.array):
        self.x = x
        return self.forward(x)

    def forward(self, x: np.array):
        B, C, H, W  = x.shape
        assert C==self.in_features
        if self.padding>0:
            x_pad = np.pad(self.x,[(0,),(0,),(self.padding,),(self.padding,)],'constant',constant_values=0)
        else:
            x_pad = self.x
        self.output_size = int((H + 2*self.padding - self.k) / self.stride + 1)
        assert self.output_size ==  ((H + 2*self.padding - self.k) / self.stride + 1)
        output = np.zeros([B,self.out_features, self.output_size,self.output_size])
        for c in range(self.out_features):
            for i in range(self.output_size):
                for j in range(self.output_size):
                    output[:,c,i,j] = np.sum(self.W_data[:,:,:,c] * x_pad[:,:, i*self.stride:self.k+i*self.stride,j*self.stride:self.k+j*self.stride],axis=(1,2,3))+self.b_data[:,c]
        # pdb.set_trace()
        return output
    def backward(self, delta: np.array):
        # pdb.set_trace()
        if self.padding>0:
            x_pad = np.pad(self.x,[(0,),(0,),(self.padding,),(self.padding,)],'constant',constant_values=0)
        else:
            x_pad = self.x
        B, C, H, W  = self.x.shape
        dx_pad = np.zeros((B,C,H+2*self.padding,W+2*self.padding))
        self.W_grad = np.zeros_like(self.W_data)
        self.b_grad = np.zeros_like(self.b_data)
        for k in range(delta.shape[0]):
            for f in range(delta.shape[1]):
                for xi in range(delta.shape[2]):
                    for xj in range(delta.shape[3]):
                        dx_pad[k,:,xi*self.stride:xi*self.stride+self.k,xj*self.stride:xj*self.stride+self.k] += self.W_data[:,:,:,f]*delta[k,f,xi,xj]
                        self.W_grad[:,:,:,f] += x_pad[k,:,xi*self.stride:xi*self.stride+self.k,xj*self.stride:xj*self.stride+self.k] * delta[k,f,xi,xj]
        dx = dx_pad[:,:,self.padding:self.padding+H,self.padding:self.padding+W]
        self.b_grad = np.sum(delta, axis=(0,2,3))
        return dx
    # usage1:
    # conv = Conv(2,0,1,1,1)
    # input = np.array([[[[1,2,3],[1,2,3],[1,2,3]]],[[[1,2,3],[1,2,3],[1,2,3]]]]) # shape = (1,1,3,3) (B,C,H,W)
    # delta = np.array([[[[1,2],[3,4]]],[[[1,2],[3,4]]]])
    # usage2:
    # conv = Conv(3,0,3,1,1)
    # input = np.ones([1,1,9,9])
    # delta = np.ones([1,1,3,3])
    # usage3: padding
    # conv = Conv(3,1,1,1,1)
    # input = np.ones([1,1,5,5])
    # delta = np.ones([1,1,5,5])
    # usage4:
    # conv = Conv(3,0,1,1,32)
    # input = np.ones([2,1,28,28])
    # delta = np.ones([2,32,26,26])

    #####
    # output = conv(input)
    # print(output.shape)
    # print(delta.shape)
    # loss = conv.backward(delta)

class MaxPooling():
    def __init__(self, kernel:int, stride:int):
        self.kernel = kernel
        self.stride = stride
        self.argmax = None
    def __call__(self, x: np.array):
        self.x = x
        return self.forward(x)
    def forward(self,x: np.array):
        self.B, self.C, H, W = x.shape
        self.output_size = int((H-self.kernel)/self.stride+1)
        out = np.zeros([self.B,self.C,self.output_size,self.output_size])
        # self.argmax = np.zeros_like(out)
        for b in range(self.B):
            for c in range(self.C):
                for i in range(self.output_size):
                    for j in range(self.output_size):
                        out[b,c,i,j] = np.max(self.x[b,c,i*self.stride:i*self.stride+self.kernel,j*self.stride:j*self.stride+self.kernel])
        # pdb.set_trace()
        return out
    def backward(self,delta: np.array):
        dx = np.zeros_like(self.x)
        for b in range(self.B):
            for c in range(self.C):
                for i in range(self.output_size):
                    for j in range(self.output_size):
                        index = np.argmax(self.x[b,c,i*self.stride:i*self.stride+self.kernel,j*self.stride:j*self.stride+self.kernel])
                        h_idx = self.stride * i + index // self.kernel
                        w_idx = self.stride * j + index % self.kernel
                        dx[b,c,h_idx,w_idx] = delta[b,c,i,j]
        # pdb.set_trace()
        return dx

    # maxpool = MaxPooling(2,2)
    # # input = np.ones([1,1,4,4])
    # # input = np.array([[[[10,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7]]]])
    # input = np.ones([4,64,11,11])
    # delta = np.ones([1,1,2,2])
    # output = maxpool(input)
    # maxpool.backward(delta)


class BatchNorm():
    def __init__(self, num_features, momentum=0.9):
        self.bn_param = defaultdict()
        self.bn_param['running_mean'], self.bn_param['running_var'] = 0, 0
        self.gamma_data = np.ones(shape=(num_features, ))
        self.beta_data = np.zeros(shape=(num_features, ))
        self.momentum = momentum
        self.gamma_velocity = np.zeros_like(self.gamma_data)
        self.beta_velocity = np.zeros_like(self.beta_data)

    def __call__(self, x: np.array, is_training=True):
        if is_training:
            return self.BatchNorm_for_train(x)
        else:
            return self.BatchNorm_for_test(x)

    def BatchNorm_for_train(self, x):
        running_mean = self.bn_param['running_mean']
        running_var = self.bn_param['running_var']
        results = 0
        x_mean = x.mean(axis=0)
        x_var = x.var(axis=0)
        self.x_norm = (x-x_mean)/np.sqrt(x_var+1e-6)
        self.x_minus_mean = x-x_mean
        self.x_var = x_var + 1e-6
        results = self.gamma_data * self.x_norm + self.beta_data
        running_mean = self.momentum * \
            running_mean + (1-self.momentum) * x_mean
        running_var = self.momentum * running_var + (1-self.momentum) * x_var
        self.bn_param['running_mean'] = running_mean
        self.bn_param['running_var'] = running_var
        return results

    def BatchNorm_for_test(self, x):
        running_mean = self.bn_param['running_mean']
        running_var = self.bn_param['running_var']
        results = 0
        x_norm = (x-running_mean)/np.sqrt(running_var+1e-6)
        results = self.gamma_data * x_norm + self.beta_data
        return results

    def backward(self, delta: np.array):
        N, D = delta.shape
        self.gamma_grad = np.sum(self.x_norm * delta, axis=0)
        self.beta_grad = np.sum(delta, axis=0)
        dx_norm = delta * self.gamma_data # (B,C) * (C,) => (B,C) 
        dsigma = -0.5 * np.sum(dx_norm * self.x_minus_mean,
                               axis=0) * np.power(self.x_var, -1.5)
        dmu = -np.sum(dx_norm / np.sqrt(self.x_var), axis=0) - \
            2 * dsigma * np.sum(self.x_minus_mean, axis=0) / N
        x_grad = dx_norm / np.sqrt(self.x_var) + 2 * \
            dsigma * (self.x_minus_mean) / N + dmu / N
        # pdb.set_trace()
        return x_grad


class SGD():
    def __call__(self, layer: FCLayer, learning_rate: float):
        self.learning_rate = learning_rate
        self.data_grad_dict = defaultdict(dict)
        for k, v in layer.__dict__.items():
            if k.endswith("_data") or k.endswith("_grad"):
                param, func = k.split("_")
                self.data_grad_dict[param][func] = v
        return self.step()

    def step(self):
        for param in self.data_grad_dict:
            if self.data_grad_dict[param]['data'] is None or self.data_grad_dict[param]['grad'] is None:
                print("No data or grad")
                continue
            self.data_grad_dict[param]['data'] -= self.learning_rate * \
                self.data_grad_dict[param]['grad']


class Momentum():
    def __init__(self, momentum: float):
        self.momentum = momentum

    def __call__(self, layer: FCLayer, learning_rate: float):
        self.learning_rate = learning_rate
        self.data_grad_dict = defaultdict(dict)
        for k, v in layer.__dict__.items():
            if k.endswith("_data") or k.endswith("_grad") or k.endswith("_velocity"):
                param, func = k.split("_")
                self.data_grad_dict[param][func] = v
        return self.step()

    def step(self):
        for param in self.data_grad_dict:
            if self.data_grad_dict[param]['data'] is None or self.data_grad_dict[param]['grad'] is None:
                print("No data or grad")
                continue
            self.data_grad_dict[param]['velocity'] = self.momentum * \
                self.data_grad_dict[param]['velocity'] + \
                self.learning_rate * self.data_grad_dict[param]['grad']
            self.data_grad_dict[param]['data'] -= self.data_grad_dict[param]['velocity']


class ReLU():
    def __call__(self, x: np.array):
        return self.forward(x)

    def forward(self, x: np.array):
        self.x_ = x
        # return np.where(self.x_ > 0, self.x_, 0)
        return np.maximum(0, self.x_)

    def backward(self):
        return np.where(self.x_ > 0, 1, 0)

    # relu = ReLU()
    # # x= np.array([[1,3,-2],[-1,-1,2]])
    # x= np.ones([1,1,5,5])
    # print(x.shape)
    # y = relu(x)
    # print(y)
    # back = relu.backward()
    # print(back)


class Tanh():
    def __call__(self, x: np.array):
        return self.forward(x)

    def forward(self, x: np.array):
        self.x_ = x
        self.res = np.tanh(self.x_)
        return self.res

    def backward(self):
        return (1 - np.power(self.res, 2))

######################################
# layer_list = [2,4,2]
# FCLayer_list = []
# for i in range(layer_list):
#     # FCLayer_list.append()
#     print(i)


# x= np.random.randn(600,2)
# layer=FCLayer([2,4])
# pred=layer(x)
# print(pred.shape)
