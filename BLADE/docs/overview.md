---
id: overview
title: Overview
sidebar_position: 3
---

# Overview of BLADE

In the BLADE package, you can load the following functions and modules.

## BLADE: Core Algorithms
A class object contains core algorithms of BLADE. Users can reach internal variables (Nu, Omega, and Beta) and functions for calculating objective functions (ELBO function) and gradients with respect to the variational parameters. There is an optimization function (`BLADE.Optimize()`) for performing L-BFGS optimization. 

- `ExpF(self.Beta)`: returns a matrix containing estimated fractions of each cell type in each sample.
- `self.Nu`: contains estimated gene expression levels of each gene in each cell type for each sample.
- `numpy.mean(self.Nu,0)`: To obtain an estimated gene expression profile per cell type, we can simply take an average across the samples.

## Framework
A framework based on the BLADE class module above. Here are the inputs and outputs for using the Framework function:

### Input arguments
- **X**: A matrix contains average gene expression profiles per cell type (a signature matrix) in log-scale.
- **stdX**: A matrix contains standard deviation per gene per cell type.
- **Y**: A matrix contains bulk gene expression data in linear-scale.
- **Ind_Marker**: Index for marker genes, defaults to [True]*Ngene.
- **Alphas, Alpha0s, Kappa0s, SYs**: Hyperparameters considered in the Empirical Bayes phase.

### Output values
- **final_obj**: A final BLADE object with optimized variational parameters and hyperparameters.
- **best_obj**: The best object from Empirical Bayes step.
- **best_set**: A list contains the hyperparameters selected in the Empirical Bayes step.
- **All_out**: A list of BLADE objects from the Empirical Bayes step.
