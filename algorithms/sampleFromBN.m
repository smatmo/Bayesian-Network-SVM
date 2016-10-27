function samples = sampleFromBN(adjacency, params, numSamples)

numVar = length(adjacency);
sortIdx = topologicalSort(adjacency);
samples = zeros(numSamples,  numVar);

numVals = zeros(numVar,1);
for k = 1:numVar
    numVals(k) = size(params{k},1);
end

for n = 1:numVar
    idx = sortIdx(n);
    parents = adjacency(:,idx) == 1;
    parentIdx = subv2ind(numVals(parents), samples(:,parents));
    samples(:,idx) = numVals(idx) + 1 - sum(cumsum(params{idx}(:,parentIdx),1) > repmat(rand(1, numSamples), numVals(idx), 1),1);
end

