function params = renormalize(adjacency, params, classIdx, numVals)
%
% params = renormalize(adjacency, params, classIdx)
%
% Perform conditional distribution preserving re-normalization as described
% in
% Wettig H., Gruenwald P., Roos T., Myllymaeki, P. and Tirri H.,
% "When discriminative learning of Bayesian network parameters is easy",
% IJCAI, 2003.
%
% Robert Peharz, 2012
%

numVar = length(adjacency);

if iscell(params) && (nargin < 4 || isempty(numVals))
    numVals = zeros(numVar,1);
    for k = 1:numVar
        numVals(k) = size(params{k},1);
    end
end

if ~iscell(params)
    if (nargin < 4 || isempty(numVals))
        error('For params in vector format, numVals needs to be provided.')
    end
    %%% calculate the offset in the params-vector, corresponding to each
    %%% variable
    wOffset = zeros(numVar,1);
    for k=2:numVar
        %%% numVals(k-1) * prod(numVals(adjacency(:,k))) == CPTsize
        wOffset(k) = wOffset(k-1) + numVals(k-1) * prod(numVals(adjacency(:,k-1) == 1));
    end
end

classChildIdx = find(adjacency(classIdx,:));
nonClassChildIdx = setdiff(1:numVar, classChildIdx);
superParents = zeros(numVar, 1);

for k = 1:length(classChildIdx)
    found = false;
    parents = find(adjacency(:,classChildIdx(k)));
    for l = 1:length(parents)
        parParents = [find(adjacency(:,parents(l))); parents(l)];
        if length(intersect(parParents, parents)) == length(parents)
            found = true;
            superParents(classChildIdx(k)) = parents(l);
            break;
        end
    end
    if found == false
        error('no Wettig structure.');
    end
end

topSortIdx = topologicalSort(adjacency);
classChildIdx = topSortIdx(ismember(topSortIdx, classChildIdx));
classChildIdx = classChildIdx(end:-1:1);

for k = 1:length(classChildIdx)
    curIdx = classChildIdx(k);
    curParents = find(adjacency(:,curIdx));
    superPar = superParents(curIdx);
    parParents = find(adjacency(:,superPar));
    extraParents = setdiff(parParents, curParents);
    isExtraParent = ismember(parParents, extraParents);
    remParentsIdx = curParents ~= superPar;
    
    parState = ones(1,length(curParents));
    while 1
        thetaIdx = subv2ind(numVals(curParents), parState);
        if iscell(params)
            lse = logsumexp(params{curIdx}(:,thetaIdx));
            params{curIdx}(:,thetaIdx) = params{curIdx}(:,thetaIdx) - lse;
        else
            sIdx = wOffset(curIdx) + (thetaIdx-1)*numVals(curIdx) + 1;
            eIdx = wOffset(curIdx) + (thetaIdx-1)*numVals(curIdx) + numVals(curIdx);
            lse = logsumexp(params(sIdx:eIdx));
            params(sIdx:eIdx) = params(sIdx:eIdx) - lse;
        end
        
        parParState = ones(1,length(parParents));
        parParState(~isExtraParent) = parState(remParentsIdx);
        extraParState = ones(length(extraParents),1);
        while 1
            parParState(isExtraParent) = extraParState;
            
            thetaIdx = subv2ind(numVals(parParents), parParState);
            if iscell(params)
                params{superPar}(parState(~remParentsIdx),thetaIdx) = params{superPar}(parState(~remParentsIdx),thetaIdx) + lse;
            else
                sIdx = wOffset(superPar) + (thetaIdx-1)*numVals(superPar) + parState(~remParentsIdx);
                params(sIdx) = params(sIdx) + lse;
            end
            
            for s = 1:length(extraParState)
                if extraParState(s) == numVals(extraParents(s))
                    extraParState(s) = 1;
                    continue
                end
                extraParState(s) = extraParState(s) + 1;
                break
            end
            if all(extraParState == 1)
                break
            end
        end
        
        
        for s = 1:length(parState)
            if parState(s) == numVals(curParents(s))
                parState(s) = 1;
                continue
            end
            parState(s) = parState(s) + 1;
            break
        end
        if all(parState == 1)
            break
        end
        
    end
end

if iscell(params)
    for k = 1:length(nonClassChildIdx)
        curIdx = nonClassChildIdx(k);
        for l = 1:size(params{curIdx},2)
            params{curIdx}(:,l) = params{curIdx}(:,l) - logsumexp(params{curIdx}(:,l));
        end
    end
else
    for k = 1:length(nonClassChildIdx)
        curIdx = nonClassChildIdx(k);
        sIdx = wOffset(curIdx) + 1;
        eIdx = wOffset(curIdx) + numVals(curIdx);
        numParStates = prod(numVals(adjacency(:,k)==1));
        for l=1:numParStates
            params(sIdx:eIdx) = params(sIdx:eIdx) - logsumexp(params(sIdx:eIdx));
            sIdx = sIdx + numVals(curIdx);
        end
    end
end

