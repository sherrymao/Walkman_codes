function g = grad_nnpca(cov_mats, x)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
g = zeros(size(x));
N = size(x, 1);
for inode = 1:N
    g(inode, :) = (-cov_mats(:,:,inode)*x(inode,:)')';
end