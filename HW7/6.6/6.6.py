from __future__ import division
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import pylab


# load data
def load_data():
    data = np.array(np.genfromtxt('knn_data.csv', delimiter=','))
    x = np.reshape(data[:, 0], (np.size(data[:, 0]), 1))
    y = np.reshape(data[:, 1], (np.size(data[:, 1]), 1))
    for i in np.arange(len(data)):
        if data[i][2] == 0:
            data[i][2] = -1
    return data, x, y

def sign(x):
    if x >= 0:
        return 1
    else:
        return -1

def knn(data, x, y, k):
    temp = np.zeros((len(data), 2))
    sum = 0
    for i in np.arange(len(data)):
        temp[i][0] = (x - data[i][0]) * (x - data[i][0]) + (y - data[i][1]) * (y - data[i][1])
        temp[i][1] = data[i][2]
    temp = temp.tolist()
    temp.sort(key=lambda x: x[0])
    temp = np.asarray(temp)
    for i in range(0, k):
        sum += temp[i][1]
    res = sign(sum)
    return res

data, x, y = load_data()
N = 10000
x1 = np.random.rand(N) * 10
y1 = np.random.rand(N) * 10


for i in np.arange(len(x1)):
    # call the knn function and set the value of k equals (1, 5, 10)
    res = knn(data, x1[i], y1[i], 1)
    if res >= 0:
        plt.scatter(x1[i], y1[i], color='red')
    else:
        plt.scatter(x1[i], y1[i], color='blue')


plt.plot(x, y, 'go')
plt.xlim(0.0, 10.0)
plt.ylim(0.0, 10.0)
plt.show()

