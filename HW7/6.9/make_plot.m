function make_plot(D, MSE_train, MSE_val)

figure;
hold on;
plot(D, MSE_train, 'yv--');
plot(D, MSE_val, 'bv--');
legend('training error', 'validation error', 'Location', 'northwest');
ax = gca;
xlabel('Degree of basis');
ylabel('Error in log scale');