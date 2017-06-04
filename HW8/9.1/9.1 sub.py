from __future__ import division
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt


def compute_euclidean_distance(point, C):
    return np.sqrt(np.sum((point[0] - C[0]) ** 2 + (point[1] - C[1]) ** 2))


def assign_label_cluster(distance):
    minVal = min(distance, key=distance.get)
    return minVal


def compute_new_centroids(W, C, X, K):
    C1 = np.zeros((2, 2))
    for i in range(0, 2):
        count = 0.0
        for j in range(0, 21):
            if W[i, j] == 1:
                count = count + 1.0
                C1[ : , i] = X[ : , j] + C1[ : , i]
        C1[:, i] = C1[ : , i] / count
    return C1

    # return np.array(cluster_label + centroids) / 2


def iterate_k_means(X, C, total_iteration):
    total_points = len(X[0])
    k = len(C[0])


    for iteration in range(0, total_iteration):
        W = np.zeros((k, total_points))
        for index_point in range(0, total_points):
            distance = {}
            for index_centroid in range(0, k):
                distance[index_centroid] = compute_euclidean_distance(X[ : , index_point],
                                                                      C[ : , index_centroid])
            label = assign_label_cluster(distance)
            W[label, index_point] = 1

        if iteration != total_iteration - 1:
            C = compute_new_centroids(W, C, X, k)
    return W, C

def plot_results(X, C, W, C0):

    K = np.shape(C)[1]

    # plot original data
    fig = plt.figure(facecolor = 'white')
    ax1 = fig.add_subplot(121)
    plt.scatter(X[0,:],X[1,:], s = 50, facecolors = 'k')
    plt.title('original data')
    ax1.set_xlim(-.55, .55)
    ax1.set_ylim(-.55, .55)
    ax1.set_aspect('equal')

    plt.scatter(C0[0,0],C0[1,0],s = 100, marker=(5, 2), facecolors = 'b')
    plt.scatter(C0[0,1],C0[1,1],s = 100, marker=(5, 2), facecolors = 'r')

    # plot clustered data
    ax2 = fig.add_subplot(122)
    colors = ['b','r']

    for k in np.arange(0,K):
        ind = np.nonzero(W[k][:]==1)[0]
        plt.scatter(X[0,ind],X[1,ind],s = 50, facecolors = colors[k])
        plt.scatter(C[0,k],C[1,k], s = 100, marker=(5, 2), facecolors = colors[k])

    plt.title('clustered data')
    ax2.set_xlim(-.55, .55)
    ax2.set_ylim(-.55, .55)
    ax2.set_aspect('equal')


if __name__ == "__main__":
    X = np.array(np.genfromtxt('Kmeans_demo_data.csv', delimiter=','))
    C0 = np.array([[0, 0], [-0.5, 0.5]])
    # C1 = np.array([[0, 0.5], [0, 0]])
    # centroids = create_centroids()
    total_iteration = 200
    print(X)
    [W, C] = iterate_k_means(X, C0, total_iteration)
    print(C)
    print(W)
    plot_results(X, C, W, C0)
    plt.show()