function g=df_lrsparse(H, label, W, N, N_cache, M, lmd, alpha)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

g = zeros(N, M);
for ind = 1:N
    Wnow  = W(ind,:)';
    prob = exp(-label((ind-1)*N_cache+1:ind*N_cache).*( (H((ind-1)*N_cache+1:ind*N_cache , :) * Wnow)) );
    prob = prob ./ (1+prob);
    grad(ind,:) = (( -H((ind-1)*N_cache+1:ind*N_cache , :)' * (label((ind-1)*N_cache+1:ind*N_cache) .* prob) )/N_cache +...
        (2*alpha*lmd).*(Wnow./(1+alpha.*Wnow.^2).^2))';% lmd *  W(ind,:)')' ;
end
g = grad;
            