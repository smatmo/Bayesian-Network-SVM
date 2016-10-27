function [params, objective, info] = trainMLBNSVM(adjacency, DATA, numVals, lambda, gamma, options)
% TRAINMLBNSVM computes the parameters of a ML-BN-SVM using gradient descent
%
% Usage:
%  [params, objective, info] = trainBNSVMgd(adjacency, DATA, numVals, ...
%                                              lambda, gamma, options)
%
% adjacency: adjacency matrix of Bayesian network structure.
%            adjacency has to be of dimensions (N+1) x (N+1), where N+1 is
%            the number of random variables (class values plus N features).
%            N+1 has to be consistent with DATA and numVals.
%            The elements of adjacency have to be in {0, 1} (binary
%            matrix).
%            adjacency(i,j) == 1 <=> there existst an edge from i'th to the
%            j'th random variable.
%
% DATA: training data.
%       DATA has dimensions M x N+1, where M is the number of samples, and N+1
%       is the number of random variables. N+1 has to be consistent with
%       adjacency and numVals.
%       DATA represents discrete data. DATA(n,d) has to be integer, larger
%       or equal 1, and smaller or equal numVals(d).
%
% numVals: number of (discrete) values of the data.
%          numVals has to be a vector with N+1 entries, where N+1 is the
%          number of variables. N+1 has to be consistent with adjacency and
%          DATA.
%
% lambda: trade-off factor of the ML-BN-SVM.
%         If lambda is 0, then only the weighted l1-norm, aka negative
%         log-likelihood is minimized, i.e. the algorithm returns the
%         maximum likelihood solution.
%         When lambda goes to infinity, the sample margins are emphasized.
%         ATTENTION: very large values of lambda (> 1e6) do not work too
%         well .
%
% gamma: desired log-margin.
%
%
% Options are:
%   + alpha ... data count smoothing factor
%   + verbosity ... yes/no
%   + maxIter ... maximal number of algorithm iterations
%   + maxIterProjection ... maximal number of projection iterations
%       (for projection on the logsumexp <= 0 set)
%   + initStepsize ... stepsize will be initially set to
%       stepsize = initStepsize / norm(gradient)
%   + useCG ... yes/no
%   + precomputeDelta ... if yes, computation will be faster, but need more
%       memory
%   + stoppingCriterionMaxParameterChange ... algorithm aborts when
%       max(abs(paramsold - paramsnew))
%           <= stoppingCriterionMaxParameterChange * length(params)
%   + stoppingCriterionMaxAbsParameterChange ... algorithm aborts when
%       max(abs(paramsold - paramsnew))
%           <= stoppingCriterionMaxAbsParameterChange
%   + initParams ... starting point for optimization
%   + resetSearchDirection ... reset search direction all
%       (resetSearchDirection) iterations
%
% For more details, see 
% Robert Peharz, Sebastian Tschiatschek and Franz Pernkopf,
% The Most Generative Maximum Margin Bayesian Networks, ICML 2013.
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if (size(adjacency,1) ~= size(adjacency,2))
    error('adjacency has to be a square matrix')
end

uniAd = unique(adjacency(:));
if (length(uniAd) ~= 2) || any((uniAd ~= [0;1]))
    error('adjacency has to be binary')
end

if ~isWettigStructure(adjacency, 1)
    error('graph structure has to be C1-structure (Wettig et al. 2003)')
end

numVals = numVals(:);
if length(numVals) ~= size(adjacency,1)
    error('length(numVals) ~= length(adjacency)')
end

if length(numVals) ~= size(DATA,2)
    error('length(numVals) ~= size(DATA,2)')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin < 6 || ~isstruct(options) || ~isfield(options, 'alpha')
    options.alpha = 1;
end

if nargin < 6 || ~isstruct(options) || ~isfield(options, 'verbosity')
    options.verbosity = 0;
end

if nargin < 6 || ~isstruct(options) || ~isfield(options, 'maxIter')
    options.maxIter = 1e3;
end

if nargin < 6 || ~isstruct(options) || ~isfield(options, 'maxIterProjection')
    options.maxIterProjection = 5;
end

if nargin < 6 || ~isstruct(options) || ~isfield(options, 'initStepsize')
    options.initStepsize = 1;
end

if nargin < 6 || ~isstruct(options) || ~isfield(options, 'useCG')
    options.useCG = 1;
end

%%% if options.precomputeDelta == 0, Delta matix will be computed on the
%%% fly, which is about 5-10 times slower, but more memory efficient
if nargin < 6 || ~isstruct(options) || ~isfield(options, 'precomputeDelta')
    options.precomputeDelta = 1;
end

if nargin < 6 || ~isstruct(options) || ~isfield(options, 'stoppingCriterionMaxParameterChange')
    options.stoppingCriterionMaxParameterChange = 1e-9;
end

if nargin < 6 || ~isstruct(options) || ~isfield(options, 'stoppingCriterionMaxAbsParameterChange')
    options.stoppingCriterionMaxAbsParameterChange = 1e-3;
end

if nargin < 6 || ~isstruct(options) || ~isfield(options, 'initParams')
    options.initParams = calcMLparams(adjacency, DATA, numVals, options.alpha);
end

if nargin < 6 || ~isstruct(options) || ~isfield(options, 'resetSearchDirection')
    options.resetSearchDirection = 20;
end

if nargin < 6 || ~isstruct(options) || ~isfield(options, 'maxTime')
    options.maxTime = inf;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if lambda == 0  % solution is ML
    params = calcMLparams(adjacency, DATA, numVals, options.alpha);
    n = computeFrequencyCounts(adjacency, numVals, DATA, options.alpha);
    objective = -n' * cellParsToPars(params);
    info.objectiveIter = objective;
    return;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[~, Np1] = size(DATA);

parents = cell(Np1,1);
parentStates = zeros(Np1, 1);
for k = 1:Np1
    parents{k} = find(adjacency(:,k));
    parentStates(k) = prod(numVals(parents{k}));
end

% precompute Delta matrix
if options.precomputeDelta
    Delta = computeDeltaMatrix(adjacency, numVals, DATA);
else
    Delta = [];
end

% precompute frequency counts
n = computeFrequencyCounts(adjacency, numVals, DATA, options.alpha);

params = cellParsToPars(options.initParams);
oldgradient = [];
info.objectiveIter = [];
nrFailParamChange = 0;

%%% for line search
thetaRange = linspace(0,1,21);
thetaRange = thetaRange(2:end);

%%% for time out
startTime = clock;

for i = 1:options.maxIter
    % compute objective and gradient
    
    [objective, gradient] = MLBNSVMobj(adjacency, numVals, params, lambda, gamma, DATA, Delta, n, options);
    
    if options.verbosity
        fprintf(1, 'Norm of gradient: %f\n', norm(gradient));
    end
    
    if i == 1  % set initial stepsize
        stepsize = options.initStepsize / norm(gradient);
    end
    
    if options.useCG
        % http://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method
        if ~isempty(oldgradient)
            beta = gradient'*(gradient-oldgradient)/(oldgradient'*oldgradient); % Polak-Ribiere
            beta = max(0, beta); % direction reset automatically
            if mod(i, options.resetSearchDirection) == 0
                beta = 0;
            end
            if beta == 0 && options.verbosity
                fprintf(1, 'Search direction reseted.\n');
            end
            gradient = gradient + beta*oldgradient;
        end
        oldgradient = gradient;
    end
    
    % gradient descent
    oldparams = params;
    params = params - stepsize * gradient;
    
    % project onto the logsumexp <= 0 set
    sidx = 1;
    for k = 1:Np1
        curParents = parents{k};
        eidx = sidx + numVals(k)*parentStates(k) - 1;
        t = projectLSE0(reshape(params(sidx:eidx), numVals(k), ...
            prod(numVals(curParents))), reshape(oldparams(sidx:eidx), ...
            numVals(k), prod(numVals(curParents))), options.maxIterProjection);
        params(sidx:eidx) = t(:);
        sidx = eidx + 1;
    end
    
    oldobjective = objective;
    wRange = oldparams * (1-thetaRange) + params * thetaRange;
    
    objectiveRange = MLBNSVMobj(adjacency, numVals, wRange, lambda, gamma, DATA, Delta, n, options);
    
    objectiveRange = [oldobjective; objectiveRange];
    [objective, minIdx] = min(objectiveRange);
    if minIdx == 1
        theta = 0;
    else
        theta = thetaRange(minIdx-1);
    end
    
    if options.verbosity
        fprintf(1, 'Linesearch: optimal theta = %f (objective=%f)\n', theta, objective);
    end
    params = (1-theta)*oldparams + theta*params;
    
    % bookkeeping
    info.objectiveIter(end+1) = objective;
    
    % display performance information
    if options.verbosity
        fprintf(1, '>> Step %d: obj=%f (oldobj=%f, stepsize=%f)\n', i, objective, oldobjective, stepsize);
    end
    
    % step-size control
    if theta >= 0.9
        stepsize = stepsize * 2;
    elseif theta <= 0.1,
        stepsize = stepsize / 2;
    end
    stepsize = min(1000, stepsize);
    
    % check convergence
    pardev = max(abs(params - oldparams)); % maximal deviation of parameters
    if pardev <= options.stoppingCriterionMaxParameterChange * length(params) && pardev <= options.stoppingCriterionMaxAbsParameterChange
        nrFailParamChange = nrFailParamChange + 1;
        if options.verbosity
            fprintf(1, 'Converged. Maximal parameter deviation is %f (nrFail=%d).\n', pardev, nrFailParamChange);
        end
    else
        nrFailParamChange = 0;
        if options.verbosity
            fprintf(1, 'Not converged.  Maximal parameter deviation is %f.\n', pardev);
        end
    end
    
    if nrFailParamChange >= 3  % we failed three times to achieve a 'large' parameter change
        break;
    end
    
    if etime(clock, startTime) > options.maxTime
        if options.verbosity
            fprintf('time limit reached, break\n');
        end
        break
    end
end

% convert parameters to cell format and perform final renormalization
params = parsToCellPars(adjacency, numVals, params);
params = renormalize(adjacency, params, 1);


