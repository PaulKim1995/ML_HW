# For Kaggle submissions, using best SVM's

# MNIST best SVM had C of 10^-6, spam had C of 100

from DataLoad import *
from sklearn.svm import SVC
import numpy as np
import pandas as pd

# mnist
mnist_mac = SVC(kernel='linear', C=pow(10, -6))
mnist_train = mnist_raw[np.random.choice(mnist_train.shape[0], 10000)]
mnist_X = mnist_train[:, 0:783]
mnist_c = mnist_train[:, 784]

mnist_mac.fit(mnist_X, mnist_c)

mnist_predict = mnist_mac.predict(mnist_test[:, 0:783])

d = {
    "Id": np.arange(0, len(mnist_predict)),
    "Category": mnist_predict
}
df = pd.DataFrame(data=d)
df.to_csv("mnist_predict.csv", index=False)

# spam
spam_mac = SVC(kernel='linear', C=100)
spam_X = spam_raw[:, 0:len(spam_raw[0]) - 1]
spam_c = spam_raw[:, len(spam_raw[0]) - 1]

spam_mac.fit(spam_X, spam_c)

spam_predict = spam_mac.predict(spam_test_data)

d = {
    "Id": np.arange(0, len(spam_predict)),
    "Category": spam_predict
}
df = pd.DataFrame(data=d)
df.to_csv("spam_predict.csv", index=False)



