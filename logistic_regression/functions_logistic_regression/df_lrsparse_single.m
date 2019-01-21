function g=df_lrsparse_single(H, label, w, ind, N_cache, M, lmd, alpha)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

g = zeros( 1, M);
Wnow  = w';
prob = exp(-label((ind-1)*N_cache+1:ind*N_cache).*( (H((ind-1)*N_cache+1:ind*N_cache , :) * Wnow)) );
prob = prob ./ (1+prob);
g = (( -H((ind-1)*N_cache+1:ind*N_cache , :)' * (label((ind-1)*N_cache+1:ind*N_cache) .* prob) )/N_cache +...
    (2*alpha*lmd).*(Wnow./(1+alpha.*Wnow.^2).^2))';% lmd *  W(ind,:)')' ;