from numpy import *
import matplotlib.pyplot as plt

def load_data():
    data = matrix(genfromtxt('breast_cancer_data.csv', delimiter=','))
    X = asarray(data[:,0:8])
    y = asarray(data[:,8])
    return (X,y)

def sigmoid(x):
    return 1/(1+exp(-x))

def gradientofsoft(x,y,w):
    t=-y*dot(x.T,w)
    r=sigmoid(t)
    gradient=-dot(x,y*r)
    return gradient

def hessianofsoft(X,y,w):
    sum=[[0] for i in range(9)]
    for k in range(len(y)):
        at=X.T[k].reshape(1,9)
        a= at.T
        sum=sum+sigmoid(-y[k]*dot(at,w))*(1-sigmoid(-y[k]*dot(at,w)))*dot(a,a.T)
    return sum

def gradientofsquare(x,y,w):
    return -2*dot(x,maximum(0,1-y*dot(x.T,w))*y)

def hessianofsquare(X,y,w):
    total=[[0, 0, 0, 0, 0, 0, 0, 0, 0] for i in range(9)]
    for p in range(len(y)):
        if maximum(0,1-y[p]*dot(X.T[p],w))>0:
            at=X.T[p].reshape(1,9)
            a=X.T[p].reshape(9,1)
            total = total+a*at
    return 2*total

def newton_soft(X,y):
    steps=[]
    num_count=[]
    temp = shape(X)
    w = asarray([[0] for i in range(9)])
    temp = ones((temp[0], 1))
    X = concatenate((temp, X), 1)
    X = X.T
    grad = 1
    k = 1
    max_its = 20
    while  k <= max_its:
        grad = gradientofsoft(X,y,w)
        hess = linalg.inv(matrix(hessianofsoft(X,y,w)))
        mis_count = 0
        for p in range(len(y)):
            if maximum(0,-y[p]*dot(X.T[p],w))>0:
                mis_count+=1
        steps.append(k)
        num_count.append(mis_count)
        w= w - asarray(hess*grad)
        k+=1
    return steps,num_count

def newton_square(X,y):
    steps = []
    num_count = []
    temp = shape(X)
    w = asarray([[0] for i in range(9)])
    temp = ones((temp[0], 1))
    X = concatenate((temp, X), 1)
    X = X.T
    grad = 1
    k = 1
    max_its = 20
    while k <= max_its:
        grad = gradientofsquare(X,y,w)
        hess = linalg.inv(matrix(hessianofsquare(X,y,w)))
        mis_count = 0
        for p in range(len(y)):
            if maximum(0,-y[p]*dot(X.T[p],w))>0:
                mis_count+=1
        steps.append(k)
        num_count.append(mis_count)
        w= w - asarray(hess*grad)
        k+=1
    return steps, num_count

X, y = load_data()
print(X)
print(y)
X1, y1 =newton_square(X, y)
X2, y2 = newton_soft(X, y)
Xm1 = X1[1:20]
Xm2 = X2[1:20]
ym1 = y1[1:20]
ym2 = y2[1:20]
# Plot the data
plt.plot(Xm1, ym1, label='square')
plt.plot(Xm2, ym2, label='soft')
plt.xlabel('the number of iteration')
plt.ylabel('the number of misclassification')

# Add a legend
plt.legend()

# Show the plot
plt.show()

