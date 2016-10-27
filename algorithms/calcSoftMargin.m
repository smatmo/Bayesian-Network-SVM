function [SM, logMargins] = calcSoftMargin(adjacency, DATA, params, gamma)

logMargins = calcLogMargins(adjacency, DATA, params);
SM = sum(min(logMargins, gamma));
