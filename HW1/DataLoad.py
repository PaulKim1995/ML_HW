from scipy.io import loadmat
from numpy.random import shuffle
from numpy import concatenate
import numpy as np

# loading data from file
mnist_raw = np.array(loadmat("hw01_data/mnist/train.mat").get('trainX'))
mnist_test = np.array(loadmat("hw01_data/mnist/test.mat").get('testX'))

spam = loadmat("hw01_data/spam/spam_data.mat")
spam_train_data = spam.get('training_data')
spam_train_labels = spam.get('training_labels')
spam_test_data = spam.get('test_data')

cifar_train = loadmat("hw01_data/cifar/train.mat").get('trainX')
cifar_test = loadmat("hw01_data/cifar/test.mat").get('testX')


# Problem 1: Data Partitioning

# shuffling the data:

shuffle(mnist_raw)

spam_raw = np.array(concatenate((spam_train_data, spam_train_labels.T), axis=1))
shuffle(spam_raw)

shuffle(cifar_train)

# make training and validation sets
mnist_valid = mnist_raw[:9999]
mnist_train = mnist_raw[10000:]

length_spam = spam_raw.shape[0]
spam_valid = spam_raw[:length_spam*0.2]
spam_train = spam_raw[(length_spam*0.2) + 1:]

cifar_valid = cifar_train[:4999]
cifar_train = cifar_train[5000:]

mnist_valid = np.array(mnist_valid)
mnist_train = np.array(mnist_train)

spam_valid = np.array(spam_valid)
spam_train = np.array(spam_train)

cifar_valid = np.array(cifar_valid)
cifar_train = np.array(cifar_train)


