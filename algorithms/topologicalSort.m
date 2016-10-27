function sortIdx = topologicalSort(adjacency)

N = size(adjacency,1);

parentList = cell(N,1);
for k = 1:N
    parentList{k} = find(adjacency(:,k));
end

sortIdx = zeros(N,1);

for l = 1:N
    foundOne = 0;
    for k = setdiff(1:N, sortIdx)
        if isempty(setdiff(parentList{k}, sortIdx(1:l-1)))
            sortIdx(l) = k;
            foundOne = 1;
            break;
        end
    end
    if ~foundOne
        sortIdx = [];
        return
    end
end


