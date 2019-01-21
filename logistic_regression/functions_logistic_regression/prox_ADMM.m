function solution = prox_ADMM(U, d, N, K, M, lmd,alpha,  D, prox_val)


 w_c = zeros(N, M); maxite_c = 2000; mu_c = 0.1;
for ite_c = 1:maxite_c
    grad = df_lr(U, d, w_c, N, K, M, lmd) + (2*alpha).*(D*w_c) + prox_val;
    w_c = w_c - mu_c * grad;
end
solution = w_c;