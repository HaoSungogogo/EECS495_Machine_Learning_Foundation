# This file is associated with the book
# "Machine Learning Refined", Cambridge University Press, 2016.
# by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.
from __future__ import division
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

def recommender_helper(X, C, W, K):
    steps=5000
    a=0.0002
    b=0.02
    W = W.T

    for step in xrange(steps):
        for i in xrange(len(X)):
            for j in xrange(len(X[i])):
                if X[i][j] > 0:
                    eij = X[i][j] - numpy.dot(C[i,:],W[:,j])
                    for k in xrange(K):
                        C[i][k] = C[i][k] + a * (2 * eij * W[k][j] - b * C[i][k])
                        W[k][j] = W[k][j] + a * (2 * eij * C[i][k] - b * W[k][j])
        e = 0
        for i in xrange(len(X)):
            for j in xrange(len(X[i])):
                if X[i][j] > 0:
                    e = e + pow(X[i][j] - numpy.dot(C[i,:],W[:,j]), 2)
                    for k in xrange(K):
                        e = e + (b/2) * ( pow(C[i][k],2) + pow(W[k][j],2) )
        if e < 0.001:
            break
    return C, W.T

def matrix_complete(X, K):
    N = len(X)
    M = len(X[0])
    C = numpy.random.rand(N, K)
    W = numpy.random.rand(M, K)
    C, W = recommender_helper(X, C, W, K)
    return C, W.T

def plot_results(X, X_corrupt, C, W):

    gaps_x = np.arange(0,np.shape(X)[1])
    gaps_y = np.arange(0,np.shape(X)[0])

    # plot original matrix
    fig = plt.figure(facecolor = 'white',figsize = (30,10))
    ax1 = fig.add_subplot(131)
    plt.imshow(X,cmap = 'hot',vmin=0, vmax=20)
    plt.title('original')

    # plot corrupted matrix
    ax2 = fig.add_subplot(132)
    plt.imshow(X_corrupt,cmap = 'hot',vmin=0, vmax=20)
    plt.title('corrupted')

    # plot reconstructed matrix
    ax3 = fig.add_subplot(133)
    recon = np.dot(C,W)
    plt.imshow(recon,cmap = 'hot',vmin=0, vmax=20)
    RMSE_mat = np.sqrt(np.linalg.norm(recon - X,'fro')/np.size(X))
    title = 'RMSE-ALS = ' + str(RMSE_mat)
    plt.title(title,fontsize=10)
    
# load in data
X = np.array(np.genfromtxt('recommender_demo_data_true_matrix.csv', delimiter=','))
X_corrupt = np.array(np.genfromtxt('recommender_demo_data_dissolved_matrix.csv', delimiter=','))

K = np.linalg.matrix_rank(X)

# run ALS for matrix completion
C, W = matrix_complete(X_corrupt, K)

# plot results
plot_results(X, X_corrupt, C, W)
plt.show()

