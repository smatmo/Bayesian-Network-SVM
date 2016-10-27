function adjacency = trainMLTAN(trainData, classNode)

numVar = size(trainData,2);

numVals = zeros(1, numVar);
for k = 1:numVar
    numVals(k) = length(unique(trainData(:,k)));
end

attrIdx = [1:classNode-1,classNode+1:numVar];
CMI = zeros(length(attrIdx),length(attrIdx));
for k = 1:length(attrIdx)-1
    for l = k+1:length(attrIdx)
        CMI(k,l) = entCompute(trainData(:,[classNode, attrIdx(k)]), numVals([classNode, attrIdx(k)])) ...
            + entCompute(trainData(:,[classNode, attrIdx(l)]), numVals([classNode, attrIdx(l)])) ...
            - entCompute(trainData(:,[classNode, attrIdx(k), attrIdx(l)]), numVals([classNode, attrIdx(k), attrIdx(l)])) ...
            - entCompute(trainData(:,classNode), numVals(classNode));
    end
end
CMI = CMI + CMI';

MST = maximumSpanningTree(CMI);
MST = MST + MST';

minNumParams = inf;
for k = 1:length(attrIdx)
    numParams = 0;
    curIdx = k;
    anchestIdx = [];
    while ~isempty(curIdx)
        childIdx = setdiff(find(MST(curIdx(1), :) == 1), anchestIdx);
        numParams = numParams + sum(numVals(attrIdx(curIdx(1))) * numVals(attrIdx(childIdx)));
        anchestIdx = [anchestIdx, curIdx(1)];
        curIdx = [curIdx(2:end), childIdx];
    end
    if numParams < minNumParams
        minNumParams = numParams;
        bestK = k;
    end
end

adjacency = zeros(numVar, numVar);
adjacency(classNode, attrIdx) = 1;
curIdx = bestK;
anchestIdx = [];
while ~isempty(curIdx)
    childIdx = setdiff(find(MST(curIdx(1), :) == 1), anchestIdx);
    adjacency(attrIdx(curIdx(1)), attrIdx(childIdx)) = 1;
    anchestIdx = [anchestIdx, curIdx(1)];
    curIdx = [curIdx(2:end), childIdx];
end

