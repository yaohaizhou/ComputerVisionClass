
# TODO:
# 搭建FC层 OK
# 搭建激活函数 tanh_OK relu_OK sigmoid
# 搭建优化器 GD_OK SGD Momentum
# 搭建完整训练模型 数据集划分_OK 所有网络模型整合（前向传播+反向传播）_OK
# BatchNorm OK
# 训练可视化Loss_OK acc_OK
# 搭建config文件

import numpy as np
from Dataloader import load_mnist, normalize
from matplotlib import pyplot as plt
from Net import *
from tqdm import tqdm
import pdb
import copy
from Log import Log
###################################### Date Preparations
X_train, Y_train, X_test, Y_test = load_mnist()
X_train = normalize(X_train)
X_test = normalize(X_test)
permutation = np.random.permutation(len(X_train))
valid_size = int(0.3*len(X_train))
X_valid = X_train[permutation[:valid_size]] # (18000, 784)
Y_valid = Y_train[permutation[:valid_size]] # (18000,)
X_train = X_train[permutation[valid_size:]] # (42000, 784)
Y_train = Y_train[permutation[valid_size:]] # (42000,)

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

def predict(X_test, Y_test, FCLayer1, FCLayer2, FCLayer3, bn1, bn2, relu1, relu2):
    # pdb.set_trace()
    z1 = FCLayer1(X_test)
    z1_bn = bn1(z1, is_training=False)
    a1 = relu1(z1_bn)
    z2 = FCLayer2(a1)
    z2_bn = bn2(z2, is_training=False)
    a2 = relu2(z2_bn)
    z3 = FCLayer3(a2)
    probs = softmax(z3)
    probs = np.argmax(probs,axis=1)
    acc = sum(probs==Y_test)/len(X_test)
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
learning_rate = 0.0001
best_acc = 0
FCLayer1 = FCLayer([784,512])
FCLayer2 = FCLayer([512,256])
FCLayer3 = FCLayer([256,10])
tanh1 = Tanh()
tanh2 = Tanh()
relu1 = ReLU()
relu2 = ReLU()
sgd = SGD()
bn1 = BatchNorm(512)
bn2 = BatchNorm(256)
log = Log()
###################################### training
for i in tqdm(range(0,epochs)):
    # pdb.set_trace()
    # Forward propagation
    z1 = FCLayer1(X_train)
    z1_bn = bn1(z1)
    a1 = relu1(z1_bn)
    z2 = FCLayer2(a1)
    z2_bn = bn2(z2)
    a2 = relu2(z2_bn)
    z3 = FCLayer3(a2)
    probs = softmax(z3) # prob.shape (42000,10)
    onehot = np.eye(10)[Y_train]
    loss = -np.sum(np.log(np.sum(probs * onehot, axis=1)),axis=0)/len(X_train)
    # Backpropagation
    delta4 = probs
    delta4[range(len(X_train)), Y_train] -= 1
    backward_loss_FCLayer3 = FCLayer3.backward(delta4)
    sgd(FCLayer3, learning_rate)
    delta3 = backward_loss_FCLayer3 * relu2.backward()
    delta3_bn = bn2.backward(delta3)
    sgd(bn2, learning_rate)
    backward_loss_FCLayer2 = FCLayer2.backward(delta3_bn)
    sgd(FCLayer2, learning_rate)
    delta2 = backward_loss_FCLayer2 * relu1.backward()
    delta2_bn =  bn1.backward(delta2)
    sgd(bn1, learning_rate)
    backward_loss_FCLayer1 = FCLayer1.backward(delta2_bn)
    sgd(FCLayer1, learning_rate)
    tmp_acc = predict(X_valid, Y_valid, FCLayer1, FCLayer2, FCLayer3, bn1, bn2, relu1, relu2)
    log(i, tmp_acc, "acc")
    log(i, loss, "loss")
    if tmp_acc > best_acc:
        print("New best accuracy")
        best_acc = tmp_acc
        FCLayer1_best = copy.deepcopy(FCLayer1)
        FCLayer2_best = copy.deepcopy(FCLayer2)
        FCLayer3_best = copy.deepcopy(FCLayer3)
        bn1_best = copy.deepcopy(bn1)
        bn2_best = copy.deepcopy(bn2)
        relu1_best = copy.deepcopy(relu1)
        relu2_best = copy.deepcopy(relu2)

    # break

###################################### testing
predict(X_test, Y_test, FCLayer1_best, FCLayer2_best, FCLayer3_best, bn1_best, bn2_best, relu1_best, relu2_best)
log.plot()
