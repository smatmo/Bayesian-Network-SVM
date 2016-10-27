function [cp] = parsToCellPars(adjacency, numVals, p)
%PARSTOCELLPARS converts a parameter vector to a cell array of parameters
%
% Usage: [cp] = parsToCellPars(adjacency, numVals, p)

cp = cell(length(numVals), 1);
sidx = 1;
for k = 1:length(numVals),
    parents = find(adjacency(:,k) == 1);
    eidx = sidx + numVals(k)*prod(numVals(parents)) - 1;

    cp{k} = reshape(p(sidx:eidx), numVals(k), prod(numVals(parents)));

    sidx = eidx + 1;
end

end

