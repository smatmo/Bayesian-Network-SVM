function [objective, gradient] = MLBNSVMobj(adjacency, numVals, w, C, gamma, DATA, Delta, n, options)
%[objective, gradient] = MLBNSVMobj(adjacency, numVals, w, C, gamma, DATA, Delta, n, options)
%
% MLBNSVMobj computes the MLBNSVM objective and its gradient of the
% at the "point" specified by the parameters w, which have to be supplied
% in vector-format (not in cell format).
%
% The following parameters can be set:
%   options.LaplaceEps
%   options.r ... radius for softHinge
%   options.eta ... parameter for softmax function
%   optione.verbosity ... 0=silent, 1=verbose
%

%% Settings
nrPars = 9;

%% Parse options
if nargin <= nrPars || ~isstruct(options) || ~isfield(options, 'LaplaceEps')
    options.LaplaceEps = 1;
end

if nargin <= nrPars || ~isstruct(options) || ~isfield(options, 'verbosity')
    options.verbosity = 0;
end

if nargin <= nrPars || ~isstruct(options) || ~isfield(options, 'r')
    options.r = min([1, gamma]);
end

if nargin <= nrPars || ~isstruct(options) || ~isfield(options, 'eta')
    options.eta = -10; % default value for eta parameter of softmax function
end

if iscell(w)
    error('MLMMobj:InvalidParameters', 'Parameters have to be supplied in vector format.');
end

numVar = length(numVals);
parents = cell(numVar,1);
for k = 1:numVar
    parents{k} = find(adjacency(:,k));
end

%%% Delta precomputed
if ~isempty(Delta)
    numSamples = size(Delta, 1) / (numVals(1)-1);
    if numSamples - round(numSamples) ~= 0
        error('Delta matrix has wrong format.')
    end
    
    %% compute objective
    llr = Delta * w;
    llr = permute(reshape(llr, numVals(1) - 1, numSamples, size(w,2)), [2,1,3]);
    
    if nargout == 2,
        [smin, smingrad] = softmin(llr, options.eta);
        [shinge, shingegrad] = softHinge(gamma - smin, options.r);
    else
        smin = softmin(llr, options.eta);
        shinge = softHinge(gamma - smin, options.r);
    end
    objective = -w'*n + C * squeeze(sum(shinge,1));
    
    %% compute the gradient?
    if nargout == 2
        stime = clock;
        [row, col, val] = find(Delta); % col -> which component of the gradient, row -> which entry of smingrad
        sgrad = -repmat(shingegrad, 1, numVals(1) -1) .* smingrad;
        sgrad = sgrad';
        sgrad = sgrad(:);
        gradient = accumarray(col, sgrad(row) .* val);
        if length(gradient) < size(Delta, 2),
            gradient = vertcat(gradient, zeros(size(Delta, 2) - length(gradient), 1));
        end
        gradient = -n + C*gradient;
        
        if options.verbosity
            fprintf(1, 'Computing the gradient took %f seconds...\n', etime(clock, stime));
        end
    end
else
    %%% compute Delta on fly
    numSamples = size(DATA,1);
    numVals = numVals(:);
    
    Ptrue  = zeros(numSamples, size(w,2));
    Pwrong = zeros(numSamples, size(w,2), numVals(1)-1);
    
    for c = 1:numVals(1)-1
        replaceClass = c * ones(numSamples, 1);
        largerIdx = replaceClass >= DATA(:,1);
        replaceClass(largerIdx) = replaceClass(largerIdx) + 1;
        
        sidx = 0;
        for k = 1:numVar
            numParStates = prod(numVals(parents{k}));
            
            if c==1
                thetaIdx = subv2ind([numVals(k); numVals(parents{k})]', [DATA(:,k), DATA(:,parents{k})]);
                Ptrue = Ptrue + w(sidx + thetaIdx, :);
            end
            
            if k == 1
                thetaIdx = subv2ind([numVals(k); numVals(parents{k})]', [replaceClass, DATA(:,parents{k})]);
            else
                curParents = parents{k};
                if any(parents{k} == 1)
                    curParents = curParents(curParents ~= 1);
                    thetaIdx = subv2ind([numVals(k); numVals(curParents); numVals(1)]', [DATA(:,k), DATA(:,curParents), replaceClass]);
                else
                    thetaIdx = subv2ind([numVals(k); numVals(parents{k})]', [DATA(:,k), DATA(:,parents{k})]);
                end
            end
            Pwrong(:,:,c) = Pwrong(:,:,c) + w(sidx + thetaIdx, :);
            sidx = sidx + numVals(k) * numParStates;
        end
    end
    
    if nargout < 2
        sminVal = softmin(permute(repmat(Ptrue, [1, 1, numVals(1)-1]) - Pwrong, [1,3,2]), options.eta);
        shinge = softHinge(gamma - sminVal, options.r);
    else
        [sminVal, sminGrad]   = softmin(permute(repmat(Ptrue, [1, 1, numVals(1)-1]) - Pwrong, [1,3,2]), options.eta);
        [shinge, shingeGrad]  = softHinge(gamma - sminVal, options.r);
    end
    
    objective = -w'*n + C * squeeze(sum(shinge,1));
    
    %% compute the gradient?
    if nargout == 2
        stime = clock;
        marginGrad = zeros(size(w));
        
        sidx = 0;
        for k = 1:numVar
            numParStates = prod(numVals(parents{k}));
            CPTlength = numVals(k) * numParStates;
            varDATA = DATA(:,k);
            parDATA = DATA(:,parents{k});
            
            thetaIdx = subv2ind([numVals(k); numVals(parents{k})]', [varDATA, parDATA]);
            locGW = [accumarray(thetaIdx, -sum(sminGrad,2) .* shingeGrad); zeros(CPTlength - max(thetaIdx),1)];
            if length(locGW) < CPTlength, % for octave compatibility
                locGW = [locGW(:); zeros(CPTlength - length(locGW), 1)];
            end
            marginGrad(sidx + 1 : sidx + CPTlength) = marginGrad(sidx + 1 : sidx + CPTlength) + locGW;
            
            for c = 1:numVals(1)-1
                varDATA = DATA(:,k);
                if k == 1
                    varDATA = c * ones(size(varDATA));
                    largerIdx = c >= DATA(:,1);
                    varDATA(largerIdx) = varDATA(largerIdx) + 1;
                end
                
                if any(parents{k} == 1)
                    idx = parents{k} == 1;
                    parDATA(:,idx) = c;
                    largerIdx = c >= DATA(:,1);
                    parDATA(largerIdx, idx) = parDATA(largerIdx, idx) + 1;
                end
                
                thetaIdx = subv2ind([numVals(k); numVals(parents{k})]', [varDATA, parDATA]);
                locGW = [accumarray(thetaIdx, -sminGrad(:,c) .* shingeGrad); zeros(CPTlength - max(thetaIdx),1)];
                if length(locGW) < CPTlength, % for octave compatibility
                    locGW = [locGW(:); zeros(CPTlength - length(locGW), 1)];
                end
                marginGrad(sidx + 1 : sidx + CPTlength) = marginGrad(sidx + 1 : sidx + CPTlength) - locGW;
            end
            
            sidx = sidx + CPTlength;
        end
        
        gradient = -n + C*marginGrad;
        
        if options.verbosity,
            fprintf(1, 'Computing the gradient took %f seconds...\n', etime(clock, stime));
        end
    end
end

