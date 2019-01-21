function xnew = prox_nn_norm(x)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xnew = zeros(size(x));
indx_pos = x>0;
xnew(indx_pos) = x(indx_pos);
if norm(xnew)>1
    xnew = xnew./norm(xnew);
end