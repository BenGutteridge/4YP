# -*- coding: utf-8 -*-
"""
2D Gaussian Mixture Model (GMM) Stochstic Gradient Descent (SGD) Experiment

Created on Thu Nov 19 12:14:06 2020

@author: benpg
"""
#%% 0. Import dependencies, set up basic parameters

# External
import autograd.numpy as np
from autograd.numpy.random import multivariate_normal
from autograd import grad
import matplotlib.pyplot as plt
plt.close('all')

# Evaluations of, and samples from, q() distributions
from utils import sample_Z, sample_pi, sample_mu, sample_lambda, samples
from utils import q_Z, q_pi, q_mu, q_lambda

# Evaluations of p() distributions
from utils import p_Z_given_pi, p_pi, p_mu_given_lambda, p_lambda
from utils import p_X_given_Z_mu_lambda

# Misc
from utils import responsibility, draw_ellipse, plot_GMM, cols

#%% 1. Create a (2D) dataset and plot it

# Define GMMs for generation
centres = [np.array([0.,3.]), np.array([2.,0.])]
covs = [np.eye(2), np.array([[0.6,0.4],
                             [0.4,0.6]])]

N_total = 10  # total number of datapoints wanted
X1 = multivariate_normal(mean=centres[0],
                         cov=covs[0],
                         size=int(N_total/2))
X2 = multivariate_normal(mean=centres[1],
                         cov=covs[1],
                         size=int(N_total/2))
X = np.concatenate((X1,X2))
# GMM has true distribution of 0.5(N(30,5) + N(70,3))

# Number of samples for estimates
N = 1
# How many mixture components do we want to start with
# (should automatically reduce the contribution of unnecessary components to 0)
K = 5


#%% 2. Sample q(Z,pi,mu,lambda) for all of the latent variables

### Initialise parameters of prior distributions of q

# Dirichlet distribution over pi parameter -- alpha
alpha = np.ones(K) 
# "by symmetry we have chosen the same factor for each of the components"
# "if alpha0 is small ... influenced primarily by data rather than priors"

# Wishart distribution over precision parameters -- scale and DoF: W and nu
# Gaussian distribution over mu parameters -- inv(beta*lambda) & m
beta = np.ones(K)*10   #  not sure about this
nu = 2.     # degrees of freedom
"""'The least informative, proper Wishart prior is obtained by setting 
nu = dimensionality', from Wikipedia"""
W0 = np.eye(2)/nu  # 'interpreted as a precision matrix'
m0 = np.zeros(2)  # "typically we would choose m0=0 by symmetry"
W, m = [], []
for k in range(K):  # mean and precision needed for each mixture component
    m.append(m0)
    W.append(W0)
"a reasonable choice for V would be a prior guess for the precision / n"


# Sample the model hyperparameters
n = 0 # Taking a single point from the dataset
x = X[n]
Z, pi, mu, lam, r_nk = samples(alpha, beta, W, m, nu, x, K)

# Display first sample of variables from q()
print("\n***SAMPLES***\nmu = ", mu, 
      "\nlams = ", lam, 
      "\npi = ", pi, 
      '\nr_nk = ', r_nk,
      "\nZ = ", Z)

# Plot the initialisation of the GMM from the first samples using ellipses
title = 'Random initialisation of Gaussian mixture'
plot_GMM(X, mu, lam, pi, centres, covs, K, title)
    
#%% 4. Calculate ELBO: plug samples from q into L = E[log p] + E[log q] 

def calculate_ELBO(x,Z,pi,mu,lam,alpha,beta,W,m,nu,K,N):
    
    # 4.1. log p(X,Z,pi,mu,lambda) = 
    #   log[ p(X|Z,mu,lambda).p(Z|pi).p(pi).p(mu|lambda).p(lambda) ]
    
    Elogp = p_X_given_Z_mu_lambda(x,Z,mu,lam)*\
                p_Z_given_pi(Z, pi)*\
                    p_pi(pi, alpha)
    for k in range(K):
        Elogp = Elogp * p_mu_given_lambda(mu[k], m[k], beta[k], lam[k])*\
                p_lambda(lam[k], nu, W[k])
    Elogp = np.log(Elogp)
    
    
    # 4.2. log q(Z,pi,mu,lambda) = 
    #   log[ q(Z).q(pi).PROD_K[ q(mu|lambda).p(lambda) ] ]
        
    Elogq = q_Z(Z,x,pi,mu,lam)*q_pi(pi,alpha)
    for k in range(K):
        Elogq = Elogq * q_mu(mu[k], m[k], beta[k], lam[k])*\
                q_lambda(lam[k], nu, W[k])
    Elogq = np.log(Elogq)
    
    L = Elogp - Elogq
    # print("Loss (ELBO) = %.3f" % L)
    return L

#%% 5. Implement autograd, find gradient of L wrt all parameters:
    # alpha, beta, m and W
    # assume nu remains the same here (need a better understanding of Wishart)
    
def calculate_grad(x,Z,pi,mu,lam,alpha,beta,W,m,nu,K,N):
    L = calculate_ELBO(x,Z,pi,mu,lam,alpha,beta,W,m,nu,K,N)
    print("\n\nLOSS (ELBO) = %.3f" % L)
    
    # Doesn't work - needs fixing
    
    # Define gradient functions for each of the updated variables with autograd
    Lgrad_alpha = grad(calculate_ELBO,5)
    Lgrad_beta = grad(calculate_ELBO,6)
    Lgrad_W = grad(calculate_ELBO,7)
    Lgrad_m = grad(calculate_ELBO,8)
    
    # Find gradients wrt each variable - use in SGD update
    d_alpha = Lgrad_alpha(x,Z,pi,mu,lam,alpha,beta,W,m,nu,K,N)
    d_beta = Lgrad_beta(x,Z,pi,mu,lam,alpha,beta,W,m,nu,K,N)
    d_W = Lgrad_W(x,Z,pi,mu,lam,alpha,beta,W,m,nu,K,N)
    d_m = Lgrad_m(x,Z,pi,mu,lam,alpha,beta,W,m,nu,K,N)
    
    print("\n\n***GRADIENTS***\ndL/d_alpha = ", d_alpha)
    print("\ndL/d_beta = ", d_beta)
    print("\ndL/d_W = ", d_W)
    print("\ndL/d_m = ", d_m)
    return d_alpha, d_beta, d_W, d_m

d_alpha, d_beta, d_W, d_m = calculate_grad(x,Z,pi,mu,lam,alpha,beta,W,m,nu,K,N)


#%% 6. Update parameters by taking step, resample and repeat

step = 0.01 # can't use the same step size for everything - but for now
step_W, step_alpha, step_beta, step_m = step, step, step, step

def update(alpha, beta, W, m, d_alpha, d_beta, d_W, d_m, n):
    # Update alpha, W, beta, m
    for i in range(len(W)):
        W[i] = W[i] + step_W*d_W[i]
        alpha[i] = alpha[i] + step_alpha*d_alpha[i]
        beta[i] = beta[i] + step_beta*d_beta[i]
        m[i] = m[i] + step_m*d_m[i]

    # New datapoint (should they be sampled at random? (without replacement))
    # n += 1
    x = X[n]
        
    Z, pi, mu, lam, r_nk = samples(alpha, beta, W, m, nu, x, K)
    L = calculate_ELBO(x,Z,pi,mu,lam,alpha,beta,W,m,nu,K,N)
    if n%200 == 0:
        title = 'Update %d' % (n+1)
        plot_GMM(X, mu, lam, pi, centres, covs, K, title)
        print("\n\nLOSS (ELBO) = %.3f" % L)
    return alpha, beta, W, m, L

n_its = N_total
ELBO = np.zeros(n_its)
for j in range(n_its):
    alpha, beta, W, m, ELBO[j] = update(alpha, beta, W, m, d_alpha, 
                                        d_beta, d_W, d_m, j)
plt.figure()
plt.plot(ELBO)
plt.xlabel('Iterations')
plt.title('ELBO')

"""
TO DO:
    - fix autograd issues and get grad_L
        - Write your own versions of np functions that aren't in autograd
            - wishart DONE (but might be wrong)
            - using lists for each of the K
        - Figure out what happens with arrays
    - implement SGD
    - plot it
    - Test against batch GD
"""


