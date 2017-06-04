import csv
from pylab import *

def aslinearfit(X, Y):
    W = dot(dot(inv(dot(X.T, X)),X.T), Y)
    return W

def sigmoid(x):
    return 1 / (1 + exp(-x))

def main():
    rfile = './bacteria_data.csv'
    csvfile = open(rfile, 'rb')
    data = csv.reader(csvfile, delimiter = ',')
    X = []
    Y = []
    for i, row in enumerate(data):
        X.append(float(row[0]))
        Y.append(float(row[1]))
    X = array(X)
    Y = array(Y)
    X1 = X[:]
    Y1 = Y[:]
    one = ones(len(X))
    X = column_stack((one,X))
    Y = Y.T
    Y = log(Y/(1-Y))
    w = aslinearfit(X,Y)

    it = np.arange(0, 24)
    plt.ylabel('bacteria data')
    plt.xlabel('hour')
    plt.title('Logistic regression as a linear system')
    g = sigmoid(w[1] * it + w[0])
    plt.plot(it, g, 'k')
    plt.plot(X1, Y1, 'ro')
    plt.show()
    plt.close()
main()