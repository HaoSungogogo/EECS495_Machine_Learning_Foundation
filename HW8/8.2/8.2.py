from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import csv


def load_data():
    data = np.array(np.genfromtxt('feat_face_data.csv', delimiter=','))
    shufle = data[np.random.permutation(range(data[:,0].size)),:]
    X = shufle[:, 0 : -1]
    y = shufle[:, -1]
    print X.shape
    X = np.c_[np.ones((len(X),1)), X]
    print X
    X = X.T
    y.shape = (len(y),)

    return X, y



def Normal(X, y):
    iter = 1
    maxiter = 100
    Len = (np.linalg.norm(X)) ** 2 / 4
    alph = 1.0 / Len
    temp = np.random.rand(X[:, 1].size,) * 10 ** -3
    grad = 1
    res = np.zeros((X[:, 1].size, maxiter))

    while np.linalg.norm(grad) > 10^-14 and iter <= maxiter:
        s = 1 / (1 + np.exp(-(-y * np.matmul(X.T, temp))))
        r = -y * s
        grad = np.matmul(X, r)
        temp = temp - alph * grad
        res[:, iter - 1] = temp
        iter += 1

    return res


def Stochastic(X,y):
    iter = 1
    maxiter = 100
    temp = np.random.rand(X[:,1].size,1)* 10**-3
    grad = 1
    res = np.zeros((X[:,1].size,1))

    while np.linalg.norm(grad) > 10^-12 and iter <= maxiter:
        alph = 1.0/iter
        p = size(X[0])
        for n in range(p):
            s = -y[n] * 1/(1+np.exp(-(-y[n] * np.dot(X[:,n][newaxis] , temp)))) * X[:,n]
            s = s.T
            temp = temp - alph * s
        res = np.column_stack((res, temp))
        iter += 1
    res = res[:, 1:]
    return res


X, y = load_data()

W_normal = Normal(X, y)
W_stoch = Stochastic(X, y)

cost_normal = np.zeros(100)
for i in range(100):
    cost_normal[i] = sum(log(1 + exp(-y * np.matmul(X.T, W_normal[:, i]))))


cost_stoch = np.zeros(100)
for i in range(100):
    cost_stoch[i] = sum(log(1 + exp(-y * np.matmul(X.T, W_stoch[:, i]))))

xx = np.arange(0, 100)
plot1 = plt.plot(xx,cost_normal, label='Standard')
plot2 = plt.plot(xx,cost_stoch, label='Stochastic')
plt.xlabel('iter')
plt.ylabel('cost')
plt.legend(loc='upper left')
plt.show()
