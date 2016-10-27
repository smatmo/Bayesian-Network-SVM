function [ Delta ] = computeDeltaMatrix( adjacency, numVals, DATA )
%COMPUTEDELTAMATRIX Summary of this function goes here
%   Detailed explanation goes here

numVals = numVals(:);

classNode = 1;
numClasses = numVals(classNode);

[numSamples, numVar] = size(DATA);

parents = cell(numVar,1);
for k = 1:numVar
    parents{k} = find(adjacency(:,k));
end

lenw = 0;
for n = 1:length(numVals),
    lenw = lenw + numVals(n)*prod(numVals(adjacency(:,n) == 1));
end

nonZeros = numSamples*(numVals(classNode)-1) * numVar * 2;
rows = zeros(nonZeros, 1);
cols = zeros(nonZeros, 1);
vals = zeros(nonZeros, 1);

rowsI2 = (numClasses-1) * (0:(numSamples-1));
rowsI = 1:numSamples*(numClasses-1);

sset = zeros(numSamples, numClasses-1);
for l = 1:numSamples,
    sset(l,:) = setdiff(1:numClasses, DATA(l,classNode));
end

sidx = 1;
colOffset = 0;
for k=1:numVar
    parList = parents{k};
    parData = DATA(:, parList);
    varData = DATA(:, k);    

    % make "positive" entries
    colIndices = subv2ind([numVals(k); numVals(parList)]', horzcat(varData, parData));
    colIndices = repmat(colIndices', numClasses-1, 1);
    colIndices = colIndices(:);

    eidx = sidx + numSamples*(numClasses-1) - 1;
    rows(sidx:eidx) = rowsI;
    cols(sidx:eidx) = colOffset + colIndices;
    vals(sidx:eidx) = 1;
    sidx = sidx + numSamples*(numClasses-1);

    % make "negative" entries
    for ci = 1:(numClasses - 1),
        if k == classNode,
            varData = sset(:,ci);
        elseif any(parList == classNode),
            classNodeIdx = parList == classNode;
            parData(:,classNodeIdx) = sset(:,ci);
        end

        colIndices = subv2ind([numVals(k); numVals(parList)]', horzcat(varData, parData));

        eidx = sidx + numSamples - 1;
        rows(sidx:eidx) = rowsI2 + ci;
        cols(sidx:eidx) = colOffset + colIndices;
        vals(sidx:eidx) = -1;
        sidx = sidx + numSamples;
    end
    colOffset = colOffset + prod(numVals(parList)) * numVals(k);
end
rows(sidx:end) = [];
cols(sidx:end) = [];
vals(sidx:end) = [];

Delta = sparse(rows, cols, vals, numSamples*(numVals(classNode)-1), lenw);


end

