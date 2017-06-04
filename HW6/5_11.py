from helper import *

X, y = read_data('wavy_data.csv')
print(X)
print(y)
num_fold, num_degree = 3, 8
folds = random_split(P=y.size, K=num_fold)
# print(len(folds))
# print(folds[0])
# x1 = np.zeros((1, len(folds[0])))
# y1 = np.zeros((len(folds[0]), 1))
# j = 0
# for i in folds[0]:
#     x1[0][j] = X[0][i]
#     j = j + 1
# print("result")
# print(x1)

X_train, y_train, X_val, y_val = train_val_split(X, y, folds, fold_id=0)
# print(len(X_train[0]))
# print(len(X_val[0]))
# one = np.ones(2)
# one = np.reshape(one,(1, 2))
# zero = np.zeros((1, 2))
# print(one)
# print(zero)
# third = np.concatenate((one, zero), axis=0)
# print(third)
# F_train = fourier_basis(X_train, 3)
# print(F_train)
# F_val = fourier_basis(X_val, 3)
# print(F_val)
# w = least_square_sol(F_train, y_train)
# print("result")
# print(w)
# print("as")
# res = np.dot(F_train.T, w) - y_train
# res = res * res
# res = np.mean(res, axis= 0)
# print(res)

# res = mean_square_error(w, F_train, y_train)
# print("dfdfd")
# print(res)

MSE_train, MSE_val = [], []
D = np.arange(1, num_degree+1)
for d in D:
    F_train = fourier_basis(X_train, D=d)
    F_val = fourier_basis(X_val, D=d)
    w = least_square_sol(F_train, y_train)
    MSE_train.append(mean_square_error(w, F_train, y_train))
    MSE_val.append(mean_square_error(w, F_val, y_val))

print 'The best degree of Fourier basis, in terms of validation error, is %d' % (MSE_val.index(min(MSE_val))+1)
make_plot(D, MSE_train, MSE_val)