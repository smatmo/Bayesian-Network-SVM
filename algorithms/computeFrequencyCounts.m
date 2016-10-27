function n = computeFrequencyCounts(adjacency, numVals, DATA, LaplaceEps)
% n = computeFrequencyCounts(adjacency, numVals, DATA, LaplaceEps)

numVals = numVals(:);
[~, numVar] = size(DATA);

parents = cell(numVar,1);
for k = 1:numVar
    parents{k} = find(adjacency(:,k));
end

lenw = 0;
for n = 1:length(numVals),
    lenw = lenw + numVals(n)*prod(numVals(adjacency(:,n) == 1));
end

n = zeros(lenw,1);
sidx = 1;
for k=1:numVar,
    parList = parents{k};
    
    CPTsize = numVals(k) * prod(numVals(parList));
    eidx = sidx + CPTsize - 1;
    
    n(sidx:eidx) = histc(subv2ind([numVals(k); numVals(parList)]', horzcat(DATA(:,k), DATA(:,parList))), 1:CPTsize);
    n(sidx:eidx) = n(sidx:eidx) + LaplaceEps / CPTsize;
    
    sidx = eidx + 1;
end
