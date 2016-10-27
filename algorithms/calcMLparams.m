function [MLparams] = calcMLparams(adjacency, trainData, numVals, Laplace)
%
% [MLparams] = calcMLparams(adjacency, trainData, numVals, Laplace)
%

if Laplace < 0
    error('Laplace factor has to be nonnegative')
end

[numSamples, numVar] = size(trainData);

parents = cell(numVar,1);
for k = 1:numVar
    parents{k} = find(adjacency(:,k));
end

MLparams = cell(numVar,1);
for k = 1:numVar
    parList = parents{k};
    numParentStates = prod(numVals(parList));
    parentStates = ones(1,length(parList));
    MLparams{k} = zeros(numVals(k), numParentStates);
    
    for l = 1:numParentStates
        idx = all(trainData(:,parList) == repmat(parentStates, numSamples, 1),2);
        h = hist(trainData(idx,k), 1:numVals(k));
        
        if Laplace
            h = h + Laplace / (numVals(k) * numParentStates);
            % for octave compatibility
            if all(idx == 0), % (no instantation of the variable assignment)
                MLparams{k}(:,l) = log(1/numVals(k));
            else
                MLparams{k}(:,l) = log(h) - log(sum(h));
            end
        else
            nzIdx = h ~= 0;
            MLparams{k}(nzIdx,l) = log(h(nzIdx)) - log(sum(h(nzIdx)));
            MLparams{k}(~nzIdx,l) = -inf;
        end
        
        for m=1:length(parentStates)
            if parentStates(m) < numVals(parList(m))
                parentStates(m) = parentStates(m) + 1;
                break
            end
            parentStates(m) = 1;
        end
    end
end

