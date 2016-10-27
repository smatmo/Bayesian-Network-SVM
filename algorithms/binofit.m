function [phat, pci] = binofit(x,n,alpha)

if nargin < 3 
    alpha = 0.05;
end


% Initialize params to zero.
[row, col] = size(x);
if min(row,col) ~= 1
   error('binofit:VectorRequired');
end

[r1,c1] = size(n);
if ~isscalar(n)
   if row ~= r1 || col ~= c1
      error('binofit:InputSizeMismatch');
   end
end
if ~isfloat(x)
   x = double(x);
end

if any(n<0) || any(n~=round(n)) || any(isinf(n)) || any(x>n)
    error('binofit:InvalidN')
end
if any(x<0)
    error('binofit:InvalidX')
end
phat = x./n;

if nargout > 1
   pci=statbinoci(x,n,alpha);
end

