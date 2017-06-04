from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

### data loading
def load_data(csvname):
	data = np.array(np.genfromtxt(csvname, delimiter=','))
	X = data[:,0:-1]
	y = data[:,-1]
	y = np.reshape(y, (np.size(y), 1))
	return X, y

def sigmoid(t):
	return 1 / (1 + np.exp(-t))

def gradient_descent(X,y,M):
	b = 0
	w = np.random.rand(M, 1) * 0.01
	c = np.zeros((M, 1))
	V = np.random.rand(M, 2) * 0.01
	P = np.size(y)
	alpha = 0.01
	l_p = np.ones((P, 1))
	max_its = 10000
	k = 1
	X = X.T
	for k in range(max_its):
		q = np.zeros((P,1))
		for p in np.arange(0,P):
			x = X[p].reshape(1,np.size(X[p]))
			q[p] = sigmoid(-y[p] * (b + np.dot(w.T, np.tanh(c + np.dot(V, x.T)))))
		grad_b = -1 * np.dot(l_p.T, q * y)
		grad_w = np.zeros((M, 1))
		grad_c = np.zeros((M, 1))
		grad_V = np.zeros((M, 2))
		for m in np.arange(0, M):
			_v = V[m]
			_v.shape = (2, 1)
			t = np.tanh(c[m] +np.dot(X,_v))
			s = 1 / np.cosh(c[m]+np.dot(X,_v))**2
			grad_w[m] = -1 * np.dot(l_p.T,q * t * y)
			grad_c[m] = -1 * np.dot(l_p.T,q * s * y) * w[m]
			grad_V[m] = (-1 * np.dot(X.T, q * s * y) * w[m]).reshape(2,)
		b = b - alpha * grad_b
		w = w - alpha * grad_w
		c = c - alpha * grad_c
		V = V - alpha * grad_V
		k = k + 1
	return b, w, c, V

def plot_points(X,y):
	ind = np.nonzero(y==1)[0]
	plt.plot(X[ind,0],X[ind,1],'ro')
	ind = np.nonzero(y==-1)[0]
	plt.plot(X[ind,0],X[ind,1],'bo')

def compute_cost(c,V,t):
	F = np.tanh(c + np.dot(V,t))
	return F

def plot_separator(b,w,c,V):
	temp = np.arange(-1,1,.01)
	temp1, temp2 = np.meshgrid(temp, temp)

	temp1 = np.reshape(temp1,(np.size(temp1),1))
	temp2 = np.reshape(temp2,(np.size(temp2),1))
	g = np.zeros((np.size(temp1),1))

	t = np.zeros((2,1))
	for i in np.arange(0,np.size(temp1)):
		t[0] = temp1[i]
		t[1] = temp2[i]
		F = compute_cost(c,V,t)
		g[i] = np.tanh(b + np.dot(F.T,w))

	s1 = np.reshape(temp1, (np.size(temp),np.size(temp)))
	s2 = np.reshape(temp2, (np.size(temp),np.size(temp)))
	g = np.reshape(g,(np.size(temp),np.size(temp)))

	# plot contour in original space
	plt.contour(s1,s2,g,1,color = 'k')
	plt.gca().xaxis.set_major_locator(plt.NullLocator())
	plt.gca().yaxis.set_major_locator(plt.NullLocator())
	plt.xlim(0,1)
	plt.ylim(0,1)

# load data
X, y = load_data('genreg_data.csv')
M = 3                # number of basis functions to use / hidden units

# perform gradient descent to fit tanh basis sum
b,w,c,V = gradient_descent(X.T,y,M)

# plot resulting fit
fig = plt.figure(facecolor = 'white',figsize = (4,4))
plot_points(X,y)
plot_separator(b,w,c,V)
plt.show()
