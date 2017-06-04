function W = train(F_train, y_train, num_iter)
%Using Newton
D = size(F_train, 1); % Number of features
X = F_train;
y = y_train;
W = 0.0001 * randn(D,1);
helper = ones(1,size(X,1));
iter = 1;
grad = 1;

while iter < num_iter && norm(grad) > 10^-12
    sigma = 1./(1+exp(X'*W.*y));
    grad = X*(-sigma.*y);
    Hessian = (((sigma.*(1-sigma))*helper)'.*X)*X';
    W = W - pinv(Hessian)*grad;
    iter = iter + 1;
end

end