function [likelihood, P] = calcLikelihood(adjacency, params, DATA)
%[likelihood, P] = calcLikelihood(adjacency, params, DATA)

numVar = length(params);
numSamples = size(DATA,1);
numVals = zeros(numVar,1);
for k = 1:numVar
    numVals(k) = size(params{k},1);
end

parents = cell(numVar,1);
for k = 1:numVar
    parents{k} = find(adjacency(:,k));
end

P = zeros(numSamples, 1);
for k = 1:numVar
    varDATA = DATA(:,k);
    
    parDATA = DATA(:,parents{k});
    thetaIdx = ones(numSamples,1);
    mFact = 1;
    for l = 1:length(parents{k})
        thetaIdx = thetaIdx + mFact * (parDATA(:,l)-1);
        mFact = mFact * numVals(parents{k}(l));
    end
    
    if size(params{k},1) > 1
        P = P + params{k}(sub2ind([numVals(k), prod(numVals(parents{k}))], varDATA, thetaIdx));
    else
        P = P + params{k}(sub2ind([numVals(k), prod(numVals(parents{k}))], varDATA, thetaIdx))';
    end
end

likelihood = sum(P);
