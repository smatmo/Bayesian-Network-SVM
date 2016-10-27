function ndx = subv2ind(siz, codes)

if numel(siz) ~= size(codes,2)
    error('numel of size must be size(codes,2).')
end

if isempty(siz)
    ndx = ones(size(codes,1), 1);
    return
end

for k=1:numel(siz)
    if any(codes(:,k) > siz(k) | codes(:,k) < 1)
        error('index out ouf bounds.')
    end
end

siz = siz(:);
k = [1; cumprod(siz(1:end-1))];

ndx = 1 + (codes-1) * k;
