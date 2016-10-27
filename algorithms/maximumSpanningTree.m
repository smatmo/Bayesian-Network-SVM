function [treeAdj, objective] = maximumSpanningTree(W)
%
% find maximum weighted spanning-tree
%

[M,N] = size(W);
if M~=N
    error('weight matrix has to be quadratic.')
end
W = W - triu(inf(M));

treeAdj   = zeros(M);
reachable = eye(M);
objective = 0;

[maxVal, idx_] = max(W(:));
[idx1, idx2] = ind2sub(size(W), idx_);
while maxVal > -inf
    if ~reachable(idx1,idx2)
        objective  = objective + maxVal;
        treeAdj(idx1,idx2) = 1;
        reachable(idx1, idx2) = 1;
        reachable(idx2, idx1) = 1;
        reachable(reachable(idx1,:)==1, reachable(idx2,:)==1) = 1;
        reachable(reachable(idx2,:)==1, reachable(idx1,:)==1) = 1;
    end
    
    W(idx1,idx2) = -inf;
    [maxVal, idx_] = max(W(:));
    [idx1, idx2] = ind2sub(size(W), idx_);
    if all(reachable(:,1))
        break
    end
end
