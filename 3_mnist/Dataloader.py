import numpy as np

import sys
sys.path.append('mnist')
import mnist
# from mnist import mnist

def load_mnist():
    train_images = mnist.train_images()
    train_labels = mnist.train_labels()

    test_images = mnist.test_images()
    test_labels = mnist.test_labels()

    n_train, w, h = train_images.shape
    # print(n_train, w, h)#60000 28 28
    X_train = train_images.reshape( (n_train, w*h) )
    Y_train = train_labels

    n_test, w, h = test_images.shape
    X_test = test_images.reshape( (n_test, w*h) )
    Y_test = test_labels

    # print(X_train.shape, Y_train.shape) # 60000
    # print(X_test.shape, Y_test.shape) # 10000
    return X_train, Y_train, X_test, Y_test

def load_mnist_conv():
    train_images = mnist.train_images()
    train_labels = mnist.train_labels()

    test_images = mnist.test_images()
    test_labels = mnist.test_labels()

    n_train, w, h = train_images.shape
    X_train = train_images
    Y_train = train_labels
    n_test, w, h = test_images.shape
    X_test = test_images
    Y_test = test_labels

    X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1],X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0],1,X_test.shape[1],X_test.shape[2])
    print("Finish Dataloader")
    # print(X_train.shape, Y_train.shape) # (60000, 28, 28) (60000,)
    # print(X_test.shape, Y_test.shape) # (10000, 28, 28) (10000,)
    return X_train, Y_train, X_test, Y_test


def normalize(ori_data):
    processed_data = (ori_data-np.mean(ori_data,axis=0))/(np.std(ori_data,axis=0)+1e-6)
    # print(np.mean(processed_data[0],axis=0), np.std(processed_data[0],axis=0))
    return processed_data

# X_train, Y_train, X_test, Y_test = load_mnist_conv()
# X_train = normalize(X_train)
# print(X_train.shape)
# print(X_train)
