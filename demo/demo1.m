
clear all
close all

addpath('../algorithms')

datapath = '../data';
dataset = 'car';
load([datapath, '/', dataset, '/data.mat']);

trainData = TrainData{1};
testData = TestData{1};
nV = numVals{1};

%%% find the ML TAN structure
adjacency = trainMLTAN(trainData, 1);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
fprintf('MLBNSVM...\n');

%%% train ML-BN-SVM parameters
%%% here, lambda and gamma are set arbitrarily, and should actually 
%%% be cross-validated.
lambda = 1;
gamma = 1;
[params, objective, info] = trainMLBNSVM(adjacency, trainData, nV, lambda, gamma);

%%%
figure(1)
clf
plot(info.objectiveIter);
xlabel('#iterations')
ylabel('objective')

%%% classify test data
[predictClass, P, CR, confInt] = classify(adjacency, params, testData, 1, 0.95, nV);

%%%
LLtrain = calcLikelihood(adjacency, params, trainData);
LLtest = calcLikelihood(adjacency, params, testData);

fprintf('classification rate: %f (%f, %f)\n', CR, confInt(1), confInt(2));
fprintf('likelihood train: %f, test: %f\n', LLtrain / size(trainData,1), LLtest / size(testData,1));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
fprintf('\n')
fprintf('ML...\n');

[MLparams] = calcMLparams(adjacency, trainData, nV, 1);

%%% classify test data
[predictClass, P, CR, confInt] = classify(adjacency, MLparams, testData, 1, 0.95, nV);

%%%
LLtrain = calcLikelihood(adjacency, MLparams, trainData);
LLtest = calcLikelihood(adjacency, MLparams, testData);

fprintf('classification rate: %f (%f, %f)\n', CR, confInt(1), confInt(2));
fprintf('likelihood train: %f, test: %f\n', LLtrain / size(trainData,1), LLtest / size(testData,1));


