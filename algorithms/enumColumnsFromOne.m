function Y = enumColumnsFromOne(X)

Y = zeros(size(X));

for k=1:size(X,2)
    idx = [1:size(X,1)];
    
    l = 1;
    while any(idx)        
        idx2 = (X(idx,k) == min(X(idx,k)));        
        Y(idx(idx2), k) = l;
        idx = idx(~idx2);
        l = l + 1;
    end
end

