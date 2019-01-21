function g=df_conjugate(U, gramInv, d, y, N, m)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

g = zeros(N, m);
for ind = 1:N
    g(ind,:)= (gramInv{ind}*(y(ind, :)' + U(ind,:)'*d(ind)))';
end
            