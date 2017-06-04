from pylab import *
from numpy import *
import matplotlib.pyplot as plt

def func(y):
    z = log(1 + exp(dot(y.T,y)))
    return z
def grad(y):
    z = (2*exp(dot(y.T,y))*y)/(exp(dot(y.T,y)) + 1)
    return z
def hess(y):
    z = (2*exp(dot(y.T,y))*(2*dot(y.T,y) + exp(dot(y.T,y)) + 1))/(exp(dot(y.T,y)) + 1)**2
    return z

def hessian_descent(w0):
    max_its = 10
    iter = 0
    g_path = []
    w_path = []
    grad_eval = 0
    w_path.append(w0)
    g_path.append(func(w0))
    w = w0
    iter_path=[]
    iter_path.append(0)
    #loop
    while iter < max_its:
        grad_eval = grad(w)
        hess_eval = hess(w)
        w = w - grad_eval/hess_eval
        print "x:",w
        print "y:",func(w)
        w_path.append(w)
        g_path.append(func(w))
        iter += 1
        iter_path.append(iter)
    return w_path, g_path, iter_path

def main():
    w0 = array([1,1,1,1,1,1,1,1,1,1])
    #w0 = array([4,4,4,4,4,4,4,4,4,4])
    fig=plt.figure()
    plt.xlabel('the time of iteration')
    plt.ylabel('the value of cost function')
    fig.suptitle('the value of cost function with the change of the time of iteration')
    w_path,g_path,iter_path= hessian_descent(w0)
    plt.plot(iter_path,g_path,linewidth = 1.5,color = 'red')
    plt.legend(loc = 2)
    plt.show()
main()