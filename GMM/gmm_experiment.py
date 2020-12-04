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
from utils import sample_Z, sample_pi, sample_mu, sample_lambda
from utils import q_Z, q_pi, q_mu, q_lambda

# Evaluations of p() distributions
from utils import p_Z_given_pi, p_pi, p_mu_given_lambda, p_lambda
from utils import p_X_given_Z_mu_lambda

#Misc
from utils import responsibility, draw_ellipse, cols

#%% 1. Create a (2D) dataset and plot it

# Define GMMs for generation
centres = [np.array([0.,3.]), np.array([2.,0.])]
covs = [np.eye(2), np.array([[0.6,0.4],
                             [0.4,0.6]])]

N_total = 1000  # total number of datapoints wanted
X1 = multivariate_normal(mean=centres[0],
                         cov=covs[0],
                         size=int(N_total/2))
X2 = multivariate_normal(mean=centres[1],
                         cov=covs[1],
                         size=int(N_total/2))
X = np.concatenate((X1,X2))
plt.plot(X[:,0], X[:,1], 'kx', alpha=0.2)
# GMM has true distribution of 0.5(N(30,5) + N(70,3))

# Number of samples for estimates
N = 1
# How many mixture components do we want to start with
# (should automatically reduce the contribution of unnecessary components to 0)
K = 5


#%% 2. Sample q(Z,pi,mu,lambda) for all of the latent variables

### Initial parameters of prior distributions of q
# Dirichlet distribution over pi parameter -- alpha
alpha = np.ones(K) 
# "by symmetry we have chosen the same factor for each of the components"
# "if alpha0 is small ... influenced primarily by data rather than priors"

# Wishart distribution over precision parameters -- scale and DoF: W and nu
# Gaussian distribution over mu parameters -- inv(beta*lambda) & m
beta = np.ones(K)   #  not sure about this
nu = 2     # degrees of freedom
"""'The least informative, proper Wishart prior is obtained by setting 
nu = dimensionality', from Wikipedia"""
W0 = np.eye(2)/nu  # 'interpreted as a precision matrix'
m0 = np.zeros(2)  # "typically we would choose m0=0 by symmetry"
W, m = [], []
for k in range(K):  # mean and precision needed for each mixture component
    m.append(m0)
    W.append(W0)
"a reasonable choice for V would be a prior guess for the precision / n"

# Taking a single sample from the dataset
x = X[0]

# Sample the model hyperparameters
mu, lam = [],[]
pi = sample_pi(alpha)
for k in range(K):
    lam.append(sample_lambda(W[k], nu))
    mu.append(sample_mu(m[k], beta[k], lam[k]))

# Need to calculate responsibility of each mixture component to sample Z
r_nk = np.empty(K)
for k in range(K):
    r_nk[k] = responsibility(x,k,pi,mu,lam)
Z = sample_Z(r_nk)

# First sample of variables from q()
print("\nmu = ", mu, 
      "\nlams = ", lam, 
      "\npi = ", pi, 
      '\nr_nk = ', r_nk,
      "\nZ = ", Z)

# Plot the initialisation of the GMM from the first samples using ellipses
for k in range(K):
    x_ell, y_ell = draw_ellipse(mu[k], lam[k], cols[k])
    plt.plot(x_ell, y_ell, cols[k])
    # for whatever reason lots of the ellipses are very long and narrow, why?
    
# Plotting the ellipses for the GMM that generated the data
for i in range(2):
    x_true_ell, y_true_ell = draw_ellipse(centres[i], covs[i])
    plt.plot(x_true_ell, y_true_ell, 'g--')

plt.legend(['Datapoints',
            'k=1','k=2','k=3','k=4','k=5',
            'Data generation GMM'])

    
#%% 4. Calculate ELBO: plug samples from q into L = E[log p] + E[log q] 

def calculate_ELBO(x,Z,pi,mu,lam,alpha,beta,W,m,nu,K,N):
    
    # 4.1. log p(X,Z,pi,mu,lambda) = 
    #   log[ p(X|Z,mu,lambda).p(Z|pi).p(pi).p(mu|lambda).p(lambda) ]
    
    Elogp = p_X_given_Z_mu_lambda(x,Z,mu,lam)*\
                p_Z_given_pi(Z, pi)*\
                    p_pi(pi, alpha)
    for k in range(K):
        Elogp *= p_mu_given_lambda(mu[k], m[k], beta[k], lam[k])*\
                p_lambda(lam[k], nu, W[k])
    Elogp = np.log(Elogp)
    
    
    # 4.2. log q(Z,pi,mu,lambda) = 
    #   log[ q(Z).q(pi).PROD_K[ q(mu|lambda).p(lambda) ] ]
        
    Elogq = q_Z(Z,x,pi,mu,lam)*q_pi(pi,alpha)
    for k in range(K):
        Elogq *= q_mu(mu[k], m[k], beta[k], lam[k])*\
                q_lambda(lam[k], nu, W[k])
    Elogq = np.log(Elogq)
    
    L = Elogp - Elogq
    # print("Loss (ELBO) = %.3f" % L)
    return L


#%% 5. Implement autograd, find gradient of L wrt all parameters:
    # alpha, beta, m and W
    # assume nu remains the same here (need a better understanding of Wishart)

L = calculate_ELBO(x,Z,pi,mu,lam,alpha,beta,W,m,nu,K,N)
print("Loss = %.3f" % L)

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

print("dL/d_alpha = ", d_alpha)
print("dL/d_beta = ", d_beta)
print("dL/d_W = ", d_W)
print("dL/d_m = ", d_m)

"""
TO DO:
    - fix autograd issues and get grad_L
        - Write your own versions of np functions that aren't in autograd
        - Figure out what happens with arrays
    - implement SGD
    - plot it
    - Test against batch GD
"""


