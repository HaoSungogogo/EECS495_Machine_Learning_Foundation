function F = poly_basis(X, D)

F = ones(1, size(X,2));
for M1 = 0:D
    for M2 = 0:D
        if 0 < (M1+M2) && (M1+M2) < (D+1)
            X1 = X(1,:);
            X2 = X(2,:);
            F = [F;(X1.^M1).*(X2.^M2)];
        end
    end
end