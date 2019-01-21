function g=df_lr(H, label, W, N, N_cache, M, lmd)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

g = zeros(N, M);
for ind = 1:N
     prob = exp(-label((ind-1)*N_cache+1:ind*N_cache).*( (H((ind-1)*N_cache+1:ind*N_cache , :) * W(ind,:)')) );
    prob = prob ./ (1+prob);
    grad(ind,:) = (( -H((ind-1)*N_cache+1:ind*N_cache , :)' * (label((ind-1)*N_cache+1:ind*N_cache) .* prob) )/N_cache + lmd *  W(ind,:)')' ;
end
g = grad;
            