function xnew = prox_l1(x, beta)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xnew = zeros(size(x));
indx_pos = x>beta;
indx_neg = x<-beta;
xnew(indx_pos) = x(indx_pos) - beta;
xnew(indx_neg) = x(indx_neg) + beta;