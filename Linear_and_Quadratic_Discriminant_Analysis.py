import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
from mpl_toolkits import mplot3d
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.stats import multivariate_normal

data = load_digits().data
labels = load_digits().target

data = StandardScaler().fit_transform(data)

index = np.where((labels == 0) | (labels == 1))[0]
X = data[index, :]
Y = labels[index]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.9) 


id0 = np.where(Y_train == 0)[0]
id1 = np.where(Y_train == 1)[0]


x = [X_train[id0, :], X_train[id1, :]]
y = [Y_train[id0], Y_train[id1]]

covariance = (np.cov(x[0].T) + np.cov(x[1].T))/2

mean = [np.mean(x[0], axis=0), np.mean(x[1], axis=0)]

prior = np.array([len(y[0]), len(y[1])])/(len(y[0]) + len(y[1]))

cov_inv = np.linalg.pinv(covariance)

beta = cov_inv@(mean[1]-mean[0])
b = 0.5*(mean[0].T@cov_inv@mean[0] - mean[1].T@cov_inv@mean[1]) + np.log(prior[1]/prior[0])

prediction = np.where(X_test@beta + b>0, 1, 0)
print("predicted : ", prediction)
print("actual    : ", Y_test)

accuracy = (prediction == Y_test)
print("Accuracy is : ", np.sum(accuracy)/len(accuracy)*100)


X = data
Y = labels
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.9) 
k = np.unique(labels).shape[0]
pxl = X.shape[1]
prior = np.zeros([k, 1])
sigma = np.zeros([pxl, pxl])
mean = np.zeros([k, pxl])
samples = X_train.shape[0]


for i in range(k):
    prior[i] = len(np.where(Y_train == i)[0])/samples
    sigma += np.cov(X_train[np.where(Y_train == i)[0]].T)
    mean[i] = np.mean(X_train[np.where(Y_train == i)[0]], axis=0)
    
sigma /= k
cov_inv = np.linalg.pinv(sigma)

def classify(x):
    estimator = np.zeros([k, 1])
    for i in range(k):
        estimator[i] = -0.5*(x-mean[i]).T@cov_inv@(x-mean[i]) + np.log(prior[i])
    return np.argmax(estimator)    

predicted = np.zeros(Y_test.shape)
for i in range(Y_test.shape[0]):
    predicted[i] = classify(X_test[i])

accuracy = np.sum(predicted == Y_test)/len(Y_test)
print("Accuracy is : ", accuracy*100)

X = data
Y = labels
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.9) 
k = np.unique(labels).shape[0]
pxl = X.shape[1]
prior = np.zeros([k, 1])
sigma = np.zeros([k, pxl, pxl])
mean = np.zeros([k, pxl])
samples = X_train.shape[0]
cov_inv = np.zeros([k, pxl, pxl])

for i in range(k):
    prior[i] = len(np.where(Y_train == i)[0])/samples
    sigma[i] = np.cov(X_train[np.where(Y_train == i)[0]].T)
    cov_inv[i] = np.linalg.pinv(sigma[i])
    mean[i] = np.mean(X_train[np.where(Y_train == i)[0]], axis=0)

def classify(x):
    estimator = np.zeros([k, 1])
    for i in range(k):
        estimator[i] = -0.5*(x-mean[i]).T@cov_inv[i]@(x-mean[i]) + np.log(prior[i])
    return np.argmax(estimator)    

predicted = np.zeros(Y_test.shape)
for i in range(Y_test.shape[0]):
    predicted[i] = classify(X_test[i])
accuracy = np.sum(predicted == Y_test)/len(Y_test)
# print(Y_test)
# print(predicted)
print("Accuracy is : ", accuracy*100)  

