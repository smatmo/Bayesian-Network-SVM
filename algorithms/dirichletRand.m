function x = dirichletRand(a,N)

if ~isvector(a)
    error('a has to be a vector.')
end

if size(a,1) > size(a,2)
    transFlag = true;
else
    transFlag = false;
end

a = a(:)';
x = gamrnd(repmat(a,N,1),1,N,length(a));
x = x ./ repmat(sum(x,2), 1, length(a));

if transFlag
    x = x';
end
