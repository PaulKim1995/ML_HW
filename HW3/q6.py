import os
import numpy as np
from scipy.io import loadmat
import sklearn.preprocessing

# Loading data
mnist_raw = np.array(loadmat("hw3_mnist_dist/train.mat").get('trainX'))
mnist_test = np.array(loadmat("hw3_mnist_dist/test.mat").get('testX'))

spam = loadmat("dist/spam_data.mat")
spam_train_data = spam.get('training_data')
spam_train_labels = spam.get('training_labels')
spam_test_data = spam.get('test_data')

spam_raw = np.array(np.concatenate((spam_train_data, spam_train_labels.T), axis=1))

# Contrast-normalizing images
# print(mnist_raw.shape)
# print(mnist_raw[:, 784])

mnistX = mnist_raw[:, :784]
mnistY = mnist_raw[:, 784]

# print(mnistX[1])
mnistX = sklearn.preprocessing.normalize(mnistX)
# print(mnistX[1])

# Finding means for each digit
u_c = [np.mean(mnistX[mnistY == c]) for c in range(10)]


