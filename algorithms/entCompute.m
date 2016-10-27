function ent = entCompute(data, sz)
%
% estimate entropy of random vector (data)
%
% data: M x N, where N is the number of variables, and M the number of
% samples.
% sz: values for variable n are assumed to be in [1:sz(n)]
%

if ~isvector(sz)
    error('sz has to be a vector.');
end

if size(sz,2) > size(sz,1)
    sz = sz(:);
end

if size(data,2) ~= length(sz)
    error('size(data,2) has to be length(sz).');
end

if isempty(data)
    ent = 0;
    return
end

indices = (data-1) * [1; cumprod(sz(1:end-1))] + 1;
bins = unique(indices);
count = histc(indices, bins);
count = count / length(indices);

count = count(count > 0);
ent = -count' * log(count);
