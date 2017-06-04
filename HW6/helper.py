import numpy as np
import matplotlib.pyplot as plt

def read_data(filename):
	'''
	Read data from csvfile

	Parameters:
	----------------------------------------------------
	filename: path to data file

	Returns:
	----------------------------------------------------
	X: ndarray of shape (1, P)
	y: ndarray of shape (P, 1)
	
	Hints:
	----------------------------------------------------
	Make sure the shapes of X, y conform to above. You may
	find "np.genfromtxt(, delimiter=','), np.newaxis" useful

	'''
	####################################################
	###                  Your Code                   ###
	####################################################
	data = np.array(np.genfromtxt(filename, delimiter=','))
	X = np.reshape(data[:, 0], (1, np.size(data[:, 0])))
	y = np.reshape(data[:, 1], (np.size(data[:, 1]), 1))

	####################################################
	###                    End                       ###
	####################################################
	assert X.shape == (1, data[:, 0].size), "Shape of X incorrect"
	assert y.shape == (data[:,-1].size, 1), "Shape of y incorrect"
	return X, y

def fourier_basis(X, D):
	'''
	Return Fourier basis for X (with ONE bias dimension)

	Parameters:
	----------------------------------------------------
	X: data ndarray of shape (1, P)
	D: degree of Fourier basis features

	Returns:
	----------------------------------------------------
	F: ndarray of shape (2D+1, P)
	
	Hints:
	----------------------------------------------------
	Make sure the shapes of F conform to above. You may
	find "np.arange, np.reshape, np.concatenate(, axis=0),
	np.ones, np.cos, np.sin"  useful

	'''	
	####################################################
	###                  Your Code                   ###
	####################################################
	F = np.zeros((2 * D, len(X[0])))
	one = np.ones(len(X[0]))
	one = np.reshape(one, (1, len(X[0])))
	for i in range(0, D):
		for j in range(0, len(X[0])):
			F[2 * i][j] = np.cos(2 * np.pi * (i + 1) * X[0][j])
			F[2 * i + 1][j] = np.sin(2 * np.pi * (i + 1) * X[0][j])
	F = np.concatenate((one, F), axis=0)
	####################################################
	###                    End                       ###
	####################################################
	assert F.shape == (2*D+1, X.size), "Shape of F incorrect"
	return F

def poly_basis(X, D):
	'''
	Return polynomial basis for X (with ONE bias dimension)

	Parameters:
	----------------------------------------------------
	X: data ndarray of shape (1, P)
	D: degree of Fourier basis features

	Returns:
	----------------------------------------------------
	F: ndarray of shape (D+1, P)
	
	Hints:
	----------------------------------------------------
	Make sure the shapes of F conform to above. You may
	find "np.arange, np.reshape, np.power"  useful

	'''	
	####################################################
	###                  Your Code                   ###
	####################################################
	####################################################
	temp = np.ones((1, np.shape(X)[1]))
	F = X
	for i in range(2, D + 1):
		F = np.row_stack((F, X ** i))
	F = np.row_stack((temp, F))
####################################################
	###                    End                       ###
	####################################################
	assert F.shape == (D+1, X.size), "Shape of F incorrect"
	return F

def least_square_sol(F, y):
	'''
	Refer to eq. 5.19 in the text

	Parameters:
	----------------------------------------------------
	F: ndarray of shape (2D+1 or D+1 depends on what basis, P)
	y: ndarray of shape (P, 1)

	Returns:
	----------------------------------------------------
	w: learned weighter vector of shape (2D+1, 1)
	
	Hints:
	----------------------------------------------------
	You may find "np.linalg.pinv, np.dot"  useful

	'''	
	####################################################
	###                  Your Code                   ###
	####################################################
	# temp = np.dot(F, F.T)
	# temp2 = np.linalg.pinv(temp)
	w = np.dot(np.dot(np.linalg.pinv(np.dot(F, F.T)), F), y)

	####################################################
	###                    End                       ###
	####################################################
	assert w.shape == (F.shape[0], 1), "Shape of w incorrect"
	return w

def mean_square_error(w, F, y):
	'''
	Refer to eq. 5.19 in the text

	Parameters:
	----------------------------------------------------
	w: learned weighter vector of shape (2D+1, 1)
	F: ndarray of shape (2D+1, P)
	y: ndarray of shape (P, 1)

	Returns:
	----------------------------------------------------
	mse: a scaler, mean square error of your learned model
	
	Hints:
	----------------------------------------------------
	You may find "np.dot, np.mean"  useful

	'''	
	####################################################
	###                  Your Code                   ###
	####################################################
	res = np.dot(F.T, w) - y
	res = res * res
	mse = np.mean(res, axis=0)

	####################################################
	###                    End                       ###
	####################################################
	return mse

def random_split(P, K):
	'''
	Return a list of K arrays, each of which are indices 
	of data point
	
	Parameters:
	----------------------------------------------------
	P: number of data points
	K: number of folds

	Returns:
	----------------------------------------------------
	folds: a list of K arrays, each of which are position 
	indices of data point
	
	Hints:
	----------------------------------------------------
	You may find "np.split, np.random.permutation, np.arange" 
	useful	

	'''
	####################################################
	###                  Your Code                   ###
	####################################################
	temp = np.random.permutation(P)
	folds = np.split(temp, K)
	####################################################
	###                    End                       ###
	####################################################
	assert len(folds) == K, 'Number of folds incorrect'
	return folds

def train_val_split(X, y, folds, fold_id):
	'''
	Split the data into training and validation sets

	Parameters:
	----------------------------------------------------
	X: ndarray of shape (1, P)
	y: ndarray of shape (P, 1)
	folds: a list of K arrays, each of which are indices 
	of data point
	fold_id: the id of the fold you want to be validation set

	Returns:
	----------------------------------------------------
	X_train: training set of X
	y_train: training label
	X_val: validation set of X
	y_val: validation label

	'''
	####################################################
	###                  Your Code                   ###
	####################################################
	X_val = np.zeros((1, len(folds[fold_id])))
	y_val = np.zeros((len(folds[fold_id]), 1))
	X_train = np.zeros((1, (len(folds) - 1) * len(folds[fold_id])))
	y_train = np.zeros(((len(folds) - 1) * len(folds[fold_id]), 1))
	j = 0
	for i in folds[fold_id]:
		X_val[0][j] = X[0][i]
		y_val[j][0] = y[i][0]
		j = j + 1

	count = 0
	for k in range(0, len(folds)):
		if k != fold_id:
			for l in folds[k]:
				X_train[0][count] = X[0][l]
				y_train[count][0] = y[l][0]
				count = count + 1
	####################################################
	###                    End                       ###
	####################################################
	assert y_val.size + y_train.size == y.size, 'Split incorrect'
	# assert X_val.size + X_train.size == X.size, 'Split incorrect'
	return X_train, y_train, X_val, y_val

def make_plot(D, MSE_train, MSE_val):
	plt.figure()
	train, = plt.plot(D, MSE_train, 'yv--')
	val, = plt.plot(D, MSE_val, 'bv--')
	plt.legend(handles=[train, val], labels=['training_error', 'validation error'], loc='upper left')
	plt.xlabel('Degree of Fourier basis')
	plt.ylabel('Error in log scale')
	plt.yscale('log')
	plt.show()
