# This file is associated with the book
# "Machine Learning Refined", Cambridge University Press, 2016.
# by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

import numpy as np
import matplotlib.pyplot as plt
import csv
from math import *

from pylab import *


# sigmoid for softmax/logistic regression minimization
def sigmoid(z):
    y = 1 / (1 + np.exp(-z))
    return y


# import training data
def load_data(csvname):
    # load in data
    reader = csv.reader(open(csvname, "rb"), delimiter=",")
    d = list(reader)

    # import data and reshape appropriately
    data = np.array(d).astype("float")
    X = data[:, 0:2]
    y = data[:, 2]
    y.shape = (len(y), 1)

    # pad data with ones for more compact gradient computation
    o = np.ones((np.shape(X)[0], 1))
    X = np.concatenate((o, X), axis=1)
    X = X.T

    return X, y


def sigmoid(x):
    return 1/(1+exp(-x))

def gradientofsoft(x,y,w):
    t=-y*dot(x.T,w)
    r=sigmoid(t)
    gradient=-dot(x,y*r)
    return gradient


#YOUR CODE GOES HERE - create a gradient descent function for softmax cost/logistic regression
def softmax_grad(X, y):
    w = asarray([[0.0001] for i in range(3)])
    iter = 1
    max_its = 3000
    alpha = 0.01
    l = len(y)
    i = 0
    grad = ones((len(X), 1))
    xp = np.zeros((len(X), 1))

    while np.linalg.norm(grad) > 10**(-12) and iter <= max_its:
        # take gradient step
        #grad = - np.dot(np.dot((1 / (1 + np.exp(np.dot(np.dot(-y, X.T), w)))), X), y.T)
        #grad = - np.dot((1. / (1. + np.exp(np.dot(np.dot(y, X.T), w)))) * X, y.T)
        # grad = np.zeros((len(X),1))
        # i = 0
        # while i < l:
        #     xp[0] = 1
        #     xp[1] = X[1, i]
        #     xp[2] = X[2, i]
        #
        #     #grad = (-1 / (1 + np.exp(y[i] * np.dot(X[i].T, w)))) * y[i] * X[i] + grad
        #     grad += 1 / (1 + np.exp(y[i] * np.dot(xp.T, w))) * y[i] * xp
        #     #print(grad)
        #     i = i + 1
        grad = gradientofsoft(X,y,w)
        w = w - alpha*(grad)
        iter = iter + 1

    return w


# plots everything
def plot_all(X, y, w):
    # custom colors for plotting points
    red = [1, 0, 0.4]
    blue = [0, 0.4, 1]

    # scatter plot points
    fig = plt.figure(figsize=(4, 4))
    ind = np.argwhere(y == 1)
    ind = [s[0] for s in ind]
    plt.scatter(X[1, ind], X[2, ind], color=red, edgecolor='k', s=25)
    ind = np.argwhere(y == -1)
    ind = [s[0] for s in ind]
    plt.scatter(X[1, ind], X[2, ind], color=blue, edgecolor='k', s=25)
    plt.grid('off')

    # plot separator
    s = np.linspace(0, 1, 100)
    plt.plot(s, (-w[0] - w[1] * s) / w[2], color='k', linewidth=2)

    # clean up plot
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.show()


# load in data
# load in data
X, y = load_data('imbalanced_2class.csv')

# run gradient descent
w = softmax_grad(X, y)

# plot points and separator
plot_all(X, y, w)

