function [X_train, y_train, X_val, y_val] = train_val_split(X, y, folds, fold_id)
%   Split the data into training and validation sets
% 
% 	Parameters:
% 	----------------------------------------------------
% 	X: matrix of size (1, N)
% 	y: matrix of size (N, 1)
% 	folds: a matrix of size (K, P/K), elements at k-th
% 	        correspond to position indices in k-th fold
% 	fold_id: the id of the fold you want to be validation set
% 
% 	Returns:
% 	----------------------------------------------------
% 	X_train: training set of X
% 	y_train: training label
% 	X_val: validation set of X
% 	y_val: validation label

%% TODO
if(fold_id ~= 3)
    X_val = X(:,folds(fold_id,:));
    y_val = y(folds(fold_id,:)');
    folds(fold_id,:) = [];
    helper = reshape(folds', 1, size(folds,1)*size(folds,2));
    X_train = [X(:,helper),X(:,end)];
    y_train = [y(helper');y(end,end)];
else
    X_val = [X(:,folds(fold_id,:)),X(:,end)];
    y_val = [y(folds(fold_id,:)');y(end,end)];
    folds(fold_id,:) = [];
    helper = reshape(folds', 1, size(folds,1)*size(folds,2));
    X_train = X(:,helper);
    y_train = y(helper');
end

assert(length([y_val; y_train]) == length(y), 'Split incorrect');