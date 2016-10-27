function logMargins = calcLogMargins(adjacency, DATA, params)

classNode = 1;
numSamples = size(DATA,1);

[~, P] = classify(adjacency, params, DATA, classNode);

PtrueIdx = sub2ind([numSamples, size(P,2)], (1:numSamples)', DATA(:,classNode));
Ptrue = P(PtrueIdx);
P(PtrueIdx) = -inf;
PbestComp = max(P,[],2);

logMargins = Ptrue - PbestComp;

