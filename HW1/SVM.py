from DataLoad import *
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection


# mnist SVC
mnist_training_sizes = [100, 200, 500, 1000, 2000, 5000, 10000]
mnist_train_error = [0, 0, 0, 0, 0, 0, 0]
mnist_error = [0, 0, 0, 0, 0, 0, 0]


for i in range(0, 7):
    mnist_mac = SVC(kernel='linear')
    train = mnist_train[:mnist_training_sizes[i], 0:783]
    label = mnist_train[:mnist_training_sizes[i], 784]
    mnist_mac.fit(train, label)
    mnist_train_error[i] = 1 - mnist_mac.score(train, label)
    valid = mnist_valid[:, 0:783]
    valid_lab = mnist_valid[:, 784]
    mnist_error[i] = 1 - mnist_mac.score(valid, valid_lab)

plt.plot(mnist_training_sizes, mnist_error, 'b-')
plt.plot(mnist_training_sizes, mnist_train_error, 'r-')
plt.ylabel('Error')
plt.xlabel('Set Size')
plt.title('Mnist Error vs Training Set Size')
plt.show()


# spam SVC
spam_training_sizes = [100, 200, 500, 1000, 2000, len(spam_train)]
spam_train_error = [0, 0, 0, 0, 0, 0]
spam_error = [0, 0, 0, 0, 0, 0]


for i in range(0, 6):
    spam_mac = SVC(kernel='linear')
    train = spam_train[:spam_training_sizes[i], 0:len(spam_train[0]) - 1]
    label = spam_train[:spam_training_sizes[i], len(spam_train[0]) - 1]
    spam_mac.fit(train, label)
    spam_train_error[i] = 1 - spam_mac.score(train, label)
    valid = spam_valid[:, 0:len(spam_train[0]) - 1]
    valid_lab = spam_valid[:, len(spam_train[0]) - 1]
    spam_error[i] = 1 - spam_mac.score(valid, valid_lab)

plt.plot(spam_training_sizes, spam_error, 'b-')
plt.plot(spam_training_sizes, spam_train_error, 'r-')
plt.ylabel('Error')
plt.xlabel('Set Size')
plt.title('Spam Error vs Training Set Size')
plt.show()

# cifar SVC
cifar_training_sizes = [100, 200, 500, 1000, 2000, 5000]
cifar_train_error = [0, 0, 0, 0, 0, 0]
cifar_error = [0, 0, 0, 0, 0, 0]

for i in range(0, 6):
    cifar_mac = SVC(kernel='linear')
    train = cifar_train[:cifar_training_sizes[i], 0:len(cifar_train[0]) - 1]
    label = cifar_train[:cifar_training_sizes[i], len(cifar_train[0]) - 1]
    cifar_mac.fit(train, label)
    cifar_train_error[i] = 1 - cifar_mac.score(train, label)
    valid = cifar_valid[:, 0:len(cifar_train[0]) - 1]
    valid_lab = cifar_valid[:, len(cifar_train[0]) - 1]
    cifar_error[i] = 1 - cifar_mac.score(valid, valid_lab)

plt.plot(cifar_training_sizes, cifar_error, 'b-')
plt.plot(cifar_training_sizes, cifar_train_error, 'r-')
plt.ylabel('Error')
plt.xlabel('Set Size')
plt.title('Cifar Error vs Training Set Size')
plt.show()


# Problem 3
# C-value

error_values = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# First pass:
for i in range(0, 15):
    mnist_mac = SVC(kernel='linear', C=(pow(10, (i-7))))
    train = mnist_train[:1000, 0:783]
    label = mnist_train[:1000, 784]
    mnist_mac.fit(train, label)
    valid = mnist_valid[:, 0:783]
    valid_lab = mnist_valid[:, 784]
    error_values[i] = mnist_mac.score(valid, valid_lab)

# Ok so interesting C-values happen at the low end of the scale

# Second pass:

error_val_2 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
for i in range(0, 9):
    mnist_mac = SVC(kernel='linear', C=pow(10, (i-14)))
    train = mnist_train[:1000, 0:783]
    label = mnist_train[:1000, 784]
    mnist_mac.fit(train, label)
    valid = mnist_valid[:, 0:783]
    valid_lab = mnist_valid[:, 784]
    error_val_2[i] = mnist_mac.score(valid, valid_lab)

# 10^-5, 10^-6 and 10^-7 are the best. Now testing on 10000 data points

error_val_3 = [0, 0, 0]
for i in range(0, 3):
    mnist_mac = SVC(kernel='linear', C=pow(10, (i-7)))
    train = mnist_train[:10000, 0:783]
    label = mnist_train[:10000, 784]
    mnist_mac.fit(train, label)
    valid = mnist_valid[:, 0:783]
    valid_lab = mnist_valid[:, 784]
    error_val_3[i] = mnist_mac.score(valid, valid_lab)

# best is 10^-6.

# Problem 4
# Cross-Validation
cv_train = concatenate((spam_train, spam_valid), axis=0)  # we can use the entire set

error_values = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for i in range(0, 11):
    cv_mac = SVC(kernel='linear', C=(pow(10, (i-9))))
    train = cv_train[:, 0:len(cv_train[0]) - 1]
    label = cv_train[:, len(cv_train[0]) - 1]
    error_values[i] = np.mean(sklearn.model_selection.cross_val_score(cv_mac, train, label, cv=5))


# Anything C larger than 100 takes a prohibitive length of time, so I will use C = 100


