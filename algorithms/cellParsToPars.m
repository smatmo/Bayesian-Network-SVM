function [p] = cellParsToPars(cp)
%CELLPARSTOPARS converts a cell array of parameters to a parameter vector
%
% Usage: [p] = cellParsToPars(cp)

    p = [];
    for k = 1:length(cp),
        p = vertcat(p, cp{k}(:));
    end
end


