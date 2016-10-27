function isWettig = isWettigStructure(adjacency, classIdx)
% ISWETTIGSTRUCTURE checks for C1-Structure, aka Wettig-Structure,
% described in
%
% Wettig H., Gruenwald P., Roos T., Myllymaeki, P. and Tirri H.,
% "When discriminative learning of Bayesian network parameters is easy",
% IJCAI, 2003.
%
% Returns true, if structure is C1.
%

numVar = length(adjacency);
classChildIdx = find(adjacency(classIdx,:));
superParents = zeros(numVar, 1);

isWettig = 1;
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
        isWettig = 0;
        return;
    end
end
