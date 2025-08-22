Maxwell.py is a program that solves Maxwell equations for 2D inhomogeneous or homogeneous dielectrics using BO-SA-PINNs.

BO-SA-PINNs' core changes compared to PINNs is outlined below:

\item We propose a general multi-stage framework—BO-SA-PINNs—which uses Bayesian Optimization (BO) to automatically determine the optimal network architecture, learning rate, initial sampling points distribution and loss function weights for PINNs based on pre-training;
\item We introduce a global self-adaptive mechanism. We propose a dynamic gradient-enhanced loss balancing (DGELB) method to optimize the loss function weights and an error-based adaptive refinement with distribution (EAR-D) strategy to optimize the distribution of sampling points based on the loss and gradient information of the second stage;
\item We propose a new activation function suitable for BO-SA-PINNs;
\item We have verified the effectiveness of BO-SA-PINNs on 1D viscous Burgers equation, 1D Diffusion equation, 2D Helmholtz equation and high-dimensional Poisson equation, achieving higher accuracy with lower cost compared to existing methods.
