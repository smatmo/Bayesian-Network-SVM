# Bayesian-Network-SVM

This Matlab package implements the algorithm proposed in

Robert Peharz, Sebastian Tschiatschek, Franz Pernkopf, 
"The Most Generative Maximum Margin Bayesian Networks", 
ICML, pp. 235-243, 2013.

In particular, it trains Bayesian networks (BN) using a principled hybrid generative/discriminative objective, combining maximum likelihood (ML) and maximum margin (SVM) training, what we call ML-BN-SVM.

Run demo/demo1.m for seeing the code in action. This code simply trains a ML-BN-SVM using fixed hyper-parameters lambda and gamma. This parameters actually should be cross-validated.
