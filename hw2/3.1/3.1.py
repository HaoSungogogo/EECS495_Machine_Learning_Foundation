# import csv
# # load numpy and pandas for data manipulation
# import numpy as np
# import pandas as pd
# from pylab import *
# import matplotlib.pyplot as plt
# # load statsmodels as alias ``sm``
# import statsmodels.api as sm
#
#
#
# with open('student_debt.csv') as csvfile:
#     readCSV = csv.reader(csvfile, delimiter=',')
#     x = []
#     y = []
#     for row in readCSV:
#         year = row[0]
#         money = row[1]
#         x.append(year)
#         y.append(money)
# x1 = np.array(x)
# Y = np.array(y)
# A = np.vstack([x1, np.ones(len(x1))]).T
# m, c = np.linalg.lstsq(A, Y)[0]
# print(m, c)
# print("when the year is 2050, the answer is: ")
# print(m * 2050 + c)
# plt.axis([2003, 2015, 0, 1.5])
# plt.plot(x1, y, 'o', label='Original data', markersize=2)
# x = arange(2003,2015,0.02)
# plt.plot(x, m*x + c, 'r', label='Fitted line')
# plt.legend()
# plt.show()


import csv
from pylab import *

def linearfit(X, Y):
    W = dot(dot(inv(dot(X.T, X)),X.T), Y)
    return W



def main():
    rfile = './student_debt.csv'
    csvfile = open(rfile, 'rb')
    data = csv.reader(csvfile, delimiter = ',')
    X = []
    Y = []
    for i, row in enumerate(data):
                X.append(float(row[0]))
                Y.append(float(row[1]))
    X = array(X)
    X_temp = X[:]
    Y = array(Y)
    one = ones((len(X)))
    X = row_stack((one,X))
    X = X.T
    Y = reshape(Y, (len(Y),1))
    weight = linearfit(X,Y)
    print weight


    it = np.arange(2004, 2016, 1)
    plt.ylabel('debt')
    plt.xlabel('year')
    plt.title('linear regression')
    g = weight[1] * it + weight[0]
    plt.plot(it, g, 'k')
    plt.plot(X_temp, Y, 'ro')
    plt.show()
    plt.close()
    print weight[1] * 2050 + weight[0]
main()