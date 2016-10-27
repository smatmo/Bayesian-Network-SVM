function [Y, iterCounts] = projectLSE0(X, X0, maxIter)
%
% Y = projectLSE0(X, X0)
%
% for k = 1:size(X,2), solve
%
% minimize sum((X(:,k) - Y(:,k)).^2)
% s.t.     logsumexp(Y(:,k)) <= 0
%

[N,K] = size(X);
if N==1
    Y = zeros(1,K);
    iterCounts = zeros(1, K);
    return
end

if nargin < 2 || isempty(X0)
    X0 = log(ones(size(X))/N);
else
    X0 = X0 - repmat(logsumexp(X0), size(X0,1), 1);
end

if nargin < 3 || isempty(maxIter)
    maxIter = 500 * N;
end

LSEtol  = 1e-9/N;
IPtol   = 1e-9/N;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

notConverged = logsumexp(X, 1) > LSEtol;
X0(:, ~notConverged) = X(:, ~notConverged);

G = exp(X0(:, notConverged));
G = G ./ repmat(sqrt(sum(G.^2)), size(G,1), 1);
M = X0(:, notConverged) - G;
D = X(:, notConverged) - M;

TMP  = -M./D;
TMP(D < 0) = inf;
lambdaUB = min(TMP, [], 1);
lambdaUB(lambdaUB == inf) = 1;
lambda = 0.5 * lambdaUB;

iterCounts = zeros(1, K);
while any(notConverged)
    iterCounts(notConverged) = iterCounts(notConverged) + 1;
    bisect = true(1,sum(notConverged));
    
    while any(bisect)
        V = M(:,bisect) + D(:,bisect) .* repmat(lambda(bisect), N, 1);
        expV = exp(V);
        sumExpV = sum(expV,1);
        lambda(bisect) = lambda(bisect) - (sumExpV - 1) ./ sum(expV .* D(:,bisect),1);
        bisect(bisect) = abs(sumExpV-1) > LSEtol;
    end
    
    X0(:,notConverged) = M + D .* repmat(lambda, N, 1);
    G = exp(X0(:, notConverged));
    G = G ./ repmat(sqrt(sum(G.^2)), size(G,1), 1);
    M = X0(:, notConverged) - G;
    D = X(:, notConverged) - M;
    
    IP = sum(D .* G, 1) ./ (sqrt(sum(D.^2)));
    
    idx = IP < 1-IPtol;
    M = M(:,idx);
    D = D(:,idx);
    lambda = lambda(idx);
    notConverged(notConverged) = idx;
    
    if max(iterCounts) >= maxIter
        %fprintf('projectLSE0: maxIter reached\n');
        break
    end
end

Y = X0;


