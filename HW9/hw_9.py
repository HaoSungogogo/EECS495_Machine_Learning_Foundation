from __future__ import print_function
import numpy as np

def compactNotation(X):
	return np.hstack([np.ones([X.shape[0], 1]), X])

def readData(path):
	data = np.matrix(np.genfromtxt(path, delimiter=','))
	X = np.asarray(data[:,0:2679])
	y = np.asarray(data[:,2679])
	q = compactNotation(X)
	return (q, y)

def read_test_Data(path):
	data = np.matrix(np.genfromtxt(path, delimiter=','))
	X = np.asarray(data[:,0:2679])
	q = compactNotation(X)
	return q

# X, Y = readData('train_data_label.csv')
# print(len(X))
# print(len(X[0]))
# print(len(Y))
# print(len(Y[0]))

def sigmoid(t):
    return 1/(1 + np.exp(-t))

def checkSize(w, X, y):
	# w and y are column vector, shape [N, 1] not [N,]
	# X is a matrix where rows are data sample
	assert X.shape[0] == y.shape[0]
	assert X.shape[1] == w.shape[0]
	assert len(y.shape) == 2
	assert len(w.shape) == 2
	assert w.shape[1] == 1
	assert y.shape[1] == 1

def softmaxGrad(w, X, y):
	checkSize(w, X, y)
	X = X.T
	t = -y * np.dot(X.T, w)
	r = sigmoid(t)
	gradient = -np.dot(X, y * r)
	return gradient
	### RETURN GRADIENT

def accuracy(w, X, y):

	"""
	Calculate accuracy using matrix operations!
	"""
	l = len(y)
	y1 = np.zeros((l,1))
	y1 = np.argmax(np.dot(X, w), axis=1)
	count = 0
	for i in range(l):
		if y[i] != (y1[i] + 1):
			count = count + 1
	return 1 - (float(count)/l)

def get_label(W, X):
    l = len(X)
    Y = np.zeros((l, 1))
    Y = np.argmax(np.dot(X, W), axis=1)
    for i in range(l):
        Y[i] = Y[i] + 1
    return Y



def gradientDescent(grad, w0, X, y):
	max_iter = 5000
	alpha = 0.001
	eps = 10**(-5)

	w = w0
	iter = 0
	while True:
		gradient = grad(w, X, y)
		w = w - alpha * gradient

		if iter > max_iter or np.linalg.norm(gradient) < eps:
			break

		if iter  % 1000 == 1:
			print("Iter %d " % iter)

		iter += 1

	return w

def oneVersusAll(Y, value):
	"""
	generate label Yout, 
	where Y == value then Yout would be 1
	otherwise Yout would be -1
	"""
	l = len(Y)
	y = np.zeros((l, 1))
	for i in range(l):
		if Y[i] != value:
			y[i] = -1
		else:
			y[i] = 1
	return y



trainX, trainY = readData('train_data_label.csv')


#training individual classifier
Nfeature = trainX.shape[1]
Nclass = 4
# OVA = np.zeros((Nfeature, Nclass))
W = np.zeros((Nfeature, Nclass))
#w = np.ones(Nfeature, 1)
for i in range(Nclass):
	print("Training for class " + str(i))
	w0 = np.random.rand(Nfeature, 1)
	W[:, i:i+1] = gradientDescent(softmaxGrad, w0, trainX, oneVersusAll(trainY, (i+1)))


print("Accuracy for training set is: %f" % accuracy(W, trainX, trainY))
print(len(trainX[0]))

testX = read_test_Data('test_data.csv')
Y = get_label(W, testX)
np.savetxt("foo.csv", Y)
print(Y)