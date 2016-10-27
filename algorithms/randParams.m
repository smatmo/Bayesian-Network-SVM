function params = randParams(adjacency, numVals, alpha)
%
% params = randParams(adjacency, numVals, alpha)
%

if nargin < 3
    alpha = 1;
end

numVar = length(adjacency);

params  = cell(numVar,1);
for k = 1:numVar
    params{k} = dirichletRand(alpha * ones(numVals(k),1), prod(numVals(adjacency(:,k) == 1)));
    if numVals(k) == 1
        params{k} = params{k}';
    end
end
