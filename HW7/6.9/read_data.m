function [X, y] = read_data(filename)
% 	Read data from csvfile
% 
% 	Parameters:
% 	----------------------------------------------------
% 	filename: path to data file
% 
% 	Returns:
% 	----------------------------------------------------
% 	X: matrix of size (1, P)
% 	y: matrix of size (P, 1)
% 	
% 	Hints:
% 	----------------------------------------------------
% 	Make sure the sizes of X, y conform to above. You may
% 	find "csvread" useful

%% TODO
data = csvread(filename);
X = data(:,1:end-1)';
y = data(:,end);