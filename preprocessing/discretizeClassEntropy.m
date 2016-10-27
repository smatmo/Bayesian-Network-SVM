function [DiscData, Bins] = discretizeClassEntropy(Data, C)
%
% Discretize real-valued data for classification. This implements the
% algorithm described in 
% "Multi-Interval discretization of Continuous-Valued Attributes for 
% Classification Learning", Fayyad, U.M. and Irani, K.B., UAI, 1993.
%
% input: 
% Data: MxN matrix containing real-valued training data. Rows correspond to
% examples and columns to features.
%
% C: class values for training examples.
%
%
% output: 
% DiscData: discretized data; has the same size as Data. Discretized states
% are enumerated from 1 and the original order is preserved.
%
% Bins: cell array containing the discretization bins for each feature.
% E.g.: Bins{1} = [-inf, 3, 4.5, inf] represents the intervals [-inf, 3],
% [3, 4.5], [4.5, inf] for feature 1.
%
%
% Robert Peharz, 2012.
% 

if size(Data,1) ~= length(C)
    error('number of samples has to be equal to number of class-labels.');
end

C = C(:);

DiscData = zeros(size(Data));

for k = 1:size(Data,2)
    B = [-inf, inf];
    X = Data(:,k);
    
    while 1
        newB = [];
        for l = 1:length(B)-1
            idx = X > B(l) & X <= B(l+1);
            b = findCut(X(idx), C(idx));
            if ~isnan(b)
                newB = [newB, b];
            end
        end
        if isempty(newB)
            break
        end
        B = sort([B, newB],'ascend');
    end
    
    for l = 1:length(B)-1
        idx = X > B(l) & X <= B(l+1);
        DiscData(idx,k) = l;
    end
    Bins{k} = B;
end

end



function bestB = findCut(x, c)

if length(x) == 1
    bestB = nan;
    return
end

N    = length(c);
uniC = unique(c);
k    = length(uniC);
h    = histc(c, uniC) / N;
Ent  = -h(h ~= 0)' * log(h(h ~= 0));

uniX = unique(x);
B = [-inf; (uniX(1:end-1) + uniX(2:end)) / 2; inf];

Bind = true(size(B));
l = 1;
for r = 2:length(B) - 1
    if length(unique(c(x > B(l) & x <= B(r+1)))) == 1
        Bind(r) = false;
    else
        l = r;
    end
end
B = B(Bind);
B = B(2:end-1);

%[sortX,sortIdx] = sort(x);
%[sortX, c(sortIdx)]

bestG = -inf;
bestB = nan;
for b = B'
    c1 = c(x <= b);
    c2 = c(x > b);
    N1 = length(c1);
    N2 = length(c2);
    uniC1 = unique(c1);
    uniC2 = unique(c2);
    k1 = length(uniC1);
    k2 = length(uniC2);
    
    h1 = histc(c1, uniC1) / N1;
    h2 = histc(c2, uniC2) / N2;
    
    Ent1 = -h1(h1 ~= 0)' * log(h1(h1 ~= 0));
    Ent2 = -h2(h2 ~= 0)' * log(h2(h2 ~= 0));
    
    G = Ent - (N1 / N) * Ent1 - (N2 / N) * Ent2;
    Delta = log(3^k - 2) - k * Ent + k1 * Ent1 + k2 * Ent2;
    
    if (G > bestG) && (N * G > log(N-1) + Delta)
        bestG = G;
        bestB = b;
    end
end

end

