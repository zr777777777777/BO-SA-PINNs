Maxwell.py is a program that solves Maxwell equations for 2D inhomogeneous or homogeneous dielectrics using BO-SA-PINNs.

BO-SA-PINNs' core changes compared to PINNs is outlined below:

1. A hyperparameter search space is proposed based on various experiments and we employ Bayesian optimization to select suitable hyperparameters based on PDEs;

2. A new activation functon--TG is proposed which is suitable for PINNs;

3. We propose global self-adaptive mechanisms including EMA and RAR-D: EMA is for adjusting loss function weights and RAR-D is for adjusting sampling points distribution;

4. We have verified BO-SA-PINNs on various benchmarks including Helmholtz, Burgers and high-dimensonal Poisson, and we will share the rest of the code after the paper is published.
