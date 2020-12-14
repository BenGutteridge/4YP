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
from utils import calculate_ELBO
from utils import HiddenPrints

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
# GMM has true distribution of 0.5(N(30,5) + N(70,3))

# Number of samples for estimates
N = 1
# How many mixture components do we want to start with
# (should automatically reduce the contribution of unnecessary components to 0)
K = 5


#%% 2. Sample q(Z,pi,mu,lambda) for all of the latent variables
# Since we have constraints we use alternative parameters.
# alpha = a**2          (alpha > 0)
# W = np.dot(V.T, V)    (W positive definite)


### Initialise parameters of prior distributions of p (alpha0, beta0, m0, W0)

# Dirichlet distribution over pi parameter -- alpha
# "by symmetry we have chosen the same factor for each of the components"
# "if alpha0 is small ... influenced primarily by data rather than priors"
a0 = np.ones(K)*(0.1**0.5) # constant alpha0 for p(pi)

# Wishart distribution over precision parameters -- scale and DoF: W and nu
# Gaussian distribution over mu parameters -- inv(beta*lambda) & m
beta0 = np.ones(K)*5   #  not sure about this
nu = 2.     # degrees of freedom
"""'The least informative, proper Wishart prior is obtained by setting 
nu = dimensionality', from Wikipedia"""
V_ = np.eye(2)/nu  # 'interpreted as a precision matrix'
m_ = np.zeros(2)  # "typically we would choose m0=0 by symmetry"
V0, m0 = [], []
for k in range(K):  # mean and precision needed for each mixture component
    m0.append(m_)
    V0.append(V_)
"a reasonable choice for V would be a prior guess for the precision / n"

# Now intiialising the parameters of q (alpha, beta, m, W)
# chosen arbitrarily ? (can we improve on these?)
a = np.ones(K)
beta = np.ones(K)*10
m = [m0[k] + 1. for k in range(K)]
V = [V0[k] + np.eye(2) for k in range(K)]


# Sample the model hyperparameters
n = 0 # Taking a single point from the dataset
x = X[n]

# Transforming into parameters, thus imposing constraints
alpha = a**2
W = [np.dot(V[k].T, V[k]) for k in range(K)]

alpha0 = a0**2
W0 = [np.dot(V0[k].T, V0[k]) for k in range(K)]

p_dist_params = {\
                 'alpha': alpha0,
                 'beta': beta0,
                 'm': m0,
                 'W': W0,
                 'nu': nu,
                 }
# q_dist_params = {\
#                  'alpha': alpha,
#                  'beta': beta,
#                  'm': m,
#                  'W': W,
#                  'nu': nu
#                  }

Z, pi, mu, lam, r_nk = samples(alpha, beta, W, m, nu, x, K)

# model_params = {\
#                 'pi': pi,
#                 'mu': mu,
#                 'lam': lam,
#                 }

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


#%% 5. Implement autograd, find gradient of L wrt all parameters:
    # alpha, beta, m and W
    # assume nu remains the same here (need a better understanding of Wishart)



#%% 6. Update parameters by taking step, resample and repeat

step = 0.01 # can't use the same step size for everything - but for now

def update(x, Z, pi, mu, lam, a, beta, m, V, nu, p_dist_params, n, step, K, N):
    print('\n***Iteration %d***'%n)
    print('\nBefore update:')
    L = calculate_ELBO(x,Z,pi,mu,lam,a,beta,m,V,nu,p_dist_params,K,N)
    print("\nELBO: = %.3f" % L)
    
    # Define gradient functions for each of the updated variables with autograd
    Lgrad_a = grad(calculate_ELBO,5)
    Lgrad_beta = grad(calculate_ELBO,6)
    Lgrad_m = grad(calculate_ELBO,7)
    Lgrad_V = grad(calculate_ELBO,8)
    
    with HiddenPrints(): # suppresses print statements in calculate_ELBO
    # Find gradients wrt each variable - use in SGD update
        d_a = Lgrad_a(x,Z,pi,mu,lam,a,beta,m,V,nu,p_dist_params,K,N)
        d_beta = Lgrad_beta(x,Z,pi,mu,lam,a,beta,m,V,nu,p_dist_params,K,N)
        d_m = Lgrad_m(x,Z,pi,mu,lam,a,beta,m,V,nu,p_dist_params,K,N)
        d_V = Lgrad_V(x,Z,pi,mu,lam,a,beta,m,V,nu,p_dist_params,K,N)

    
    # print("\n\n***GRADIENTS***\ndL/d_a = ", d_a)
    # print("\ndL/d_beta = ", d_beta)
    # print("\ndL/d_m = ", d_m)
    # print("\ndL/d_V = ", d_V)
    
    # Update alpha, W, beta, m
    step_V, step_a, step_beta, step_m = step, step, step, step
    for i in range(len(V)):
        V[i] = V[i] + step_V*d_V[i]
        a[i] = a[i] + step_a*d_a[i]
        beta[i] = beta[i] + step_beta*d_beta[i]
        m[i] = m[i] + step_m*d_m[i]
    

    # New datapoint (should they be sampled at random? (without replacement))
    # n += 1
    x = X[n]
        
    alpha = a**2
    W = [np.dot(V[k].T, V[k]) for k in range(K)]
    Z, pi, mu, lam, r_nk = samples(alpha, beta, W, m, nu, x, K)

    # print("\n***SAMPLES***\nmu = ", mu, 
    #       "\nlams = ", lam, 
    #       "\npi = ", pi, 
    #       '\nr_nk = ', r_nk,
    #       "\nZ = ", Z)
    print('\nAfter update:')
    L = calculate_ELBO(x,Z,pi,mu,lam,a,beta,m,V,nu,p_dist_params,K,N)
    print("\nELBO: = %.3f" % L)
    if n%200 == 0:
        title = 'Update %d' % (n+1)
        plot_GMM(X, mu, lam, pi, centres, covs, K, title)
        print("\n\nLOSS (ELBO) = %.3f" % L)
    return alpha, beta, W, m, L

n_its = N_total
ELBO = np.zeros(n_its)
for j in range(n_its):
    alpha, beta, W, m, ELBO[j] = update(x, Z, pi, mu, lam,
                                        a, beta, m, V, nu,
                                        p_dist_params,
                                        j, step, K, N)
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
    
    - Too many function inputs - dict?
    - Sort out global variables for alpha0, etc
"""


