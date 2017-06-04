function mse = mean_square_error(w, F, y)

mse = mean(max(0,sign(-y.*(F'*w))));
