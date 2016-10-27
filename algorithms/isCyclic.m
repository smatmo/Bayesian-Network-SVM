function cyclicFlag = isCyclic(A)
% check if A represents the adjacency of a cyclic directed graph
%

cyclicFlag = any(diag(expm(A)-eye(size(A))) > 0);
