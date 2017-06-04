function folds = random_split(P, K)
%   Return a matrix of size (K, P/K), elements at k-th
% 	        correspond to position indices in k-th fold
% 	
% 	Parameters:
% 	----------------------------------------------------
% 	P: number of data points
% 	K: number of folds
% 
% 	Returns:
% 	----------------------------------------------------
% 	folds: a matrix of size (K, P/K), elements at k-th
% 	        correspond to position indices in k-th fold
%
% 	Hints:
% 	----------------------------------------------------
% 	You may find "reshape, randperm" useful	

%% assert(mod(P, K)==0, 'cannot split data into K equal folds');
%% TODO
Permutation = P - mod(P,K);
folds = reshape(randperm(Permutation), K, Permutation/K);
assert(size(folds, 1) == K, 'number of folds incorrect');
