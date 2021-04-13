from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from Net import FCLayer

def softmax(z):
    try:
        exp_scores = np.exp(z)
    except RuntimeWarning:
        import pdb;pdb.set_trace()
    probs_ = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs_

def predict(X_train, Y_train):
    probs = softmax(FCLayer1(X_train))
    probs = np.argmax(probs,axis=1)
    print("acc: ",sum(probs==Y_train)/len(Y_train))

FCLayer1 = FCLayer([2,2])
FCLayer2 = FCLayer([4,2])

X_train, Y_train=make_moons(n_samples=1000,noise=0.1)
epochs = 1000
reg_lamdba = 0
learning_rate = 0.0001

for i in tqdm(range(0,epochs)):
    # Forward propagation
    z1 = FCLayer1(X_train)
    # a1 = np.tanh(z1)
    # z2 = FCLayer2(a1)
    # import pdb;pdb.set_trace()
    probs = softmax(z1)
    # Backpropagation
    delta3 = probs
    delta3[range(len(X_train)), Y_train] -= 1
    # # dW2 = (a1.T).dot(delta3)
    # # db2 = np.sum(delta3, axis=0, keepdims=True)
    FCLayer1.backward(delta3, learning_rate)
    predict(X_train, Y_train)