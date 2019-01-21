function solution = logistic_regression_solver(U, d, prox_val, beta);

[K, M] = size(U);
w_c0 = zeros(M,1); w_c = w_c0; maxite_c = 10000; mu_c = 0.05;%0.1;
for ite_c = 1:maxite_c
    prob = exp(-d.* (U * w_c) );
    prob = prob ./ (1+prob);
    grad = ( -U' * (d.* prob) )/(K) + beta *  (w_c - prox_val);
    w_c = w_c - mu_c * grad;
end
solution = w_c';
