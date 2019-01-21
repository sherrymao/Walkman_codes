function g=df(U, d, x, N, m, lmd)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

g = zeros(N, m);
for ind = 1:N
    g(ind,:)=(U(ind,:)'*(U(ind,:)*x(ind,:)'-d(ind)))' + lmd.* x(ind, :);
end
            