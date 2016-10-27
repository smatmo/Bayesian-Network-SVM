function [minval, gradient] = softmin(X, eta)
%SOFTMIN computes the softmin.
%  For vectors, SOFTMIN(X) computes the softmin of the elements in X. For
%  matrices, SOFTMIN(X) is a row vector containing the softmin for each
%  row.
%
%  By using SOFTMIN(X, ETA) the parameters of the softmin function can
%  be changed. In detail, ETA sets eta (eta <= -1, for eta >= 1 the
%  function becomes a softmax function).
%
%  The softmin smin of x1,...,xk is defined as
%     smin = 1/eta * log(sum_i exp(eta*xi)).
%
%  R. Peharz, S. Tschiatschek, Aug. 2012

%if isvector(X),
%    X = X(:)';
%end

if nargin < 2 || isempty(eta)
    eta = -10;
end

minval = 1/eta * logsumexp(X*eta, 2);

if nargout == 2
    mX = min(X, [], 2);
    X = X - repmat(mX, 1, size(X,2));
    
    X = exp(eta * X);
    gradient = X ./ repmat(sum(X,2), 1, size(X, 2));
end
