
# TODO:
# 搭建FC层 OK
# 搭建激活函数 tanh_OK relu_OK sigmoid
# 搭建优化器 GD_OK SGD Momentum
# 搭建完整训练模型 数据集划分_OK batch_OK
# BatchNorm OK
# 训练可视化Loss_OK acc_OK
# 搭建config文件
# 封装 所有网络层封装nn模块，统一前向传播和反向传播

import numpy as np
from Dataloader import load_mnist_conv, normalize
from matplotlib import pyplot as plt
from Net import *
from tqdm import tqdm
import pdb
import copy
from Log import Log
###################################### Date Preparations
X_train, Y_train, X_test, Y_test = load_mnist_conv()
X_train = normalize(X_train)
X_test = normalize(X_test)
permutation = np.random.permutation(len(X_train))
valid_size = int(0.3*len(X_train))
X_valid = X_train[permutation[:valid_size]] # (18000, 784)
Y_valid = Y_train[permutation[:valid_size]] # (18000,)
X_train = X_train[permutation[valid_size:]] # (42000, 784)
Y_train = Y_train[permutation[valid_size:]] # (42000,)
X_train = X_train[:40] # (42000, 784)
Y_train = Y_train[:40] # (42000, 784)
X_valid = X_valid[:20] # (42000, 784)
Y_valid = Y_valid[:20] # (42000, 784)

def iterate_train(X_train, Y_train, batch_size=16):
    '''
    Usage:
        for (batch_x, batch_y) in iterate_train(X_train, Y_train, batch_size=16)
    '''
    total_seqs = X_train.shape[0]
    total_batches = total_seqs // batch_size
    for i in range(total_batches):
        start = i * batch_size
        end = start + batch_size
        batch_x = X_train[start:end]
        batch_y = Y_train[start:end]
        yield (batch_x, batch_y)
###################################### Function
def visualize(X_test, Y_test, probs, idx=0):
    '''
    plot a random figure of MNIST dataset
    Usage: 
    Visualize(27)
    '''
    plt.imshow(X_test[idx].reshape(28,28), cmap='Greys')
    print("label: ", Y_test[idx])
    print("pred: ", probs[idx])
    # plt.show()
    plt.savefig('fig.jpg')

def predict(X_valid, Y_valid, FCLayer1, FCLayer2, bn1, relu1, relu2, relu3, relu4, conv1, conv2, conv3):
    # pdb.set_trace()
    B, C, H, W  = X_valid.shape
    batch_x = conv1(X_valid)
    batch_x = relu1(batch_x)
    batch_x = maxpool1(batch_x)
    batch_x = conv2(batch_x)
    batch_x = relu2(batch_x)
    batch_x = maxpool2(batch_x)
    batch_x = conv3(batch_x)
    batch_x = relu3(batch_x)
    batch_x = batch_x.reshape(B,-1)
    z1 = FCLayer1(batch_x)
    z1_bn = bn1(z1)
    a1 = relu4(z1_bn)
    z2 = FCLayer2(a1)
    probs = softmax(z2)
    probs = np.argmax(probs,axis=1)
    acc = sum(probs==Y_valid)/len(X_valid)
    print("acc: %.2f" %(acc*100))
    return acc

def softmax(z):
    z -= np.max(z, axis = 1, keepdims = True)
    exp_scores = np.exp(z)
    probs_ = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs_

###################################### Net & hyperparameter
epochs = 100
reg_lamdba = 0
learning_rate = 0.001
best_acc = 0
FCLayer1 = FCLayer([576,64])
FCLayer2 = FCLayer([64,10])
conv1 = Conv(3,0,1,1,32)
conv2 = Conv(3,0,1,32,64)
conv3 = Conv(3,0,1,64,64)
relu1 = ReLU()
relu2 = ReLU()
relu3 = ReLU()
relu4 = ReLU()
sgd = SGD()
bn1 = BatchNorm(64)
bn2 = BatchNorm(256)
maxpool1 = MaxPooling(2,2)
maxpool2 = MaxPooling(2,2)
log = Log()
###################################### training
for i in tqdm(range(0,epochs)):
# for i in range(0,epochs):
    losses = 0
    for (batch_x, batch_y) in (iterate_train(X_train, Y_train, batch_size=20)):
        # Forward propagation
        # pdb.set_trace()
        B, C, H, W  = batch_x.shape
        batch_x = conv1(batch_x)
        batch_x = relu1(batch_x)
        batch_x = maxpool1(batch_x)
        batch_x = conv2(batch_x)
        batch_x = relu2(batch_x)
        batch_x = maxpool2(batch_x)
        batch_x = conv3(batch_x)
        batch_x = relu3(batch_x)
        batch_x = batch_x.reshape(B,-1)
        z1 = FCLayer1(batch_x)
        z1_bn = bn1(z1)
        a1 = relu4(z1_bn)
        z2 = FCLayer2(a1)
        probs = softmax(z2) # prob.shape (42000,10)
        onehot = np.eye(10)[batch_y]
        loss = -np.sum(np.log(np.sum(probs * onehot, axis=1)),axis=0)/len(batch_x)
        losses += loss
        # Backpropagation
        delta4 = probs
        delta4[range(len(batch_x)), batch_y] -= 1 # softmax backward
        backward_loss_FCLayer2 = FCLayer2.backward(delta4)
        delta3 = backward_loss_FCLayer2 * relu4.backward()
        delta3_bn = bn1.backward(delta3)
        backward_loss_FCLayer1 = FCLayer1.backward(delta3_bn)
        # pdb.set_trace()
        backward_loss_FCLayer1 = backward_loss_FCLayer1.reshape(B,64,3,3)
        delta2_relu = backward_loss_FCLayer1 * relu3.backward()
        delta2_conv = conv3.backward(delta2_relu)
        delta1 = maxpool2.backward(delta2_conv)
        delta1_relu = delta1 * relu2.backward()
        delta1_conv = conv2.backward(delta1_relu)
        delta0 = maxpool1.backward(delta1_conv)
        delta0_relu = delta0 * relu1.backward()
        delta0_conv = conv1.backward(delta0_relu)
        sgd(FCLayer2, learning_rate)
        sgd(bn1, learning_rate)
        sgd(FCLayer1, learning_rate)
        sgd(conv3, learning_rate)
        sgd(conv2, learning_rate)
        sgd(conv1, learning_rate)


    tmp_acc = predict(X_valid, Y_valid, FCLayer1, FCLayer2, bn1, relu1, relu2, relu3, relu4, conv1, conv2, conv3)
    print("losses: ",losses)
    log(i, tmp_acc, "acc")
    log(i, losses, "loss")
    if tmp_acc > best_acc:
        print("New best accuracy")
        best_acc = tmp_acc
        # FCLayer1_best = copy.deepcopy(FCLayer1)
        # FCLayer2_best = copy.deepcopy(FCLayer2)
        # bn1_best = copy.deepcopy(bn1)
        # bn2_best = copy.deepcopy(bn2)
        # relu1_best = copy.deepcopy(relu1)
        # relu2_best = copy.deepcopy(relu2)
        # conv_best1 = copy.deepcopy(conv1)
        # conv_best2 = copy.deepcopy(conv2)
        # conv_best3 = copy.deepcopy(conv3)
    # break

###################################### testing
# predict(X_test, Y_test, FCLayer1_best, FCLayer2_best, FCLayer3_best, bn1_best, bn2_best, relu1_best, relu2_best, conv_best)
log.plot()
