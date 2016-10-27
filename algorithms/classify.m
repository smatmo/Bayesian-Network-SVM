function [predictClass, P, CR, confInt] = classify(adjacency, params, DATA, classIdx, conf, numVals)
%
% [predictClass, P, CR, confInt] = classify(adjacency, params, DATA, classIdx, conf, numVals)
%
if ~iscell(params) && (nargin < 6 || isempty(numVals))
    error('for vector-params, numVals has to be provided');
end
if size(adjacency,1) ~= size(adjacency,2)
    error('adjacency is not a square matrix.');
end
if size(adjacency,1) ~= size(DATA,2)
    error('input sizes not consistent.');
end
if isCyclic(adjacency)
    error('adjacency is cyclic.');
end

if nargin < 4 || isempty(classIdx)
    classIdx = 1;
end

if nargin < 5 || isempty(conf)
    conf = 0.95;
end

[numSamples,  numVar] = size(DATA);

if iscell(params) && (nargin < 6 || isempty(numVals))
    numVals = zeros(numVar,1);
    for k = 1:numVar
        numVals(k) = size(params{k},1);
    end
end

parents = cell(numVar,1);
for k = 1:numVar
    parents{k} = find(adjacency(:,k));
end

P = zeros(numSamples, numVals(classIdx));

sidx = 0;
for k = 1:numVar
    numParStates = prod(numVals(parents{k}));
    for c = 1:numVals(classIdx)
        varDATA = DATA(:,k);
        if k == classIdx
            varDATA = c * ones(size(varDATA));
        end
        
        parSamples = DATA(:,parents{k});
        if any(parents{k} == classIdx)
            idx = parents{k} == classIdx;
            parSamples(:,idx) = c;
        end
        
        thetaIdx = subv2ind(numVals(parents{k}), parSamples);
        
        if iscell(params)
            if size(params{k},1) > 1
                P(:,c) = P(:,c) + params{k}(sub2ind([numVals(k), prod(numVals(parents{k}))], varDATA, thetaIdx));
            else
                if isscalar(params{k})
                    P(:,c) = P(:,c) + params{k};
                else
                    P(:,c) = P(:,c) + params{k}(sub2ind([numVals(k), prod(numVals(parents{k}))], varDATA, thetaIdx))';
                end
            end
        else
            thetaIdx = sub2ind([numVals(k), numParStates], varDATA, thetaIdx);
            P(:,c) = P(:,c) + params(sidx + thetaIdx);
        end
    end
    sidx = sidx + numVals(k) * numParStates;
end

[~, predictClass] = max(P,[],2);
% determine classification rate and confidence intervals
[CR, confInt] = binofit(sum(predictClass == DATA(:,1)), size(DATA, 1), 1 - conf);


