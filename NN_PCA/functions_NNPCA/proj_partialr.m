function d = proj_partialr(w, x)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% w is the argument of the projector
% x labels where the partial r is taken
d = zeros(size(x));
indx_pos = x>0;
indx_nonpos = x<=0;
d(indx_pos) = dot(w(indx_pos), x(indx_pos))/norm(x)^2.*x(indx_pos);
d(indx_nonpos)= min(w(indx_nonpos), 0);