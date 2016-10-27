function exppar = expParams(params)

exppar = cell(size(params));
for k=1:length(params)
    exppar{k} = exp(params{k});
end
