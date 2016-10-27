function logpar = logParams(params)

logpar = cell(size(params));
for k=1:length(params)
    logpar{k} = log(params{k});
end
