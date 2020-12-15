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

# Misc
from utils import samples, plot_GMM, calculate_ELBO, HiddenPrints

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

# Number of samples for estimates
N = 1
# How many mixture components do we want to start with
# (should automatically reduce the contribution of unnecessary components to 0)
K = 5
# Number of  updates to perform
n_its = 10


#%% 2. Initialise parameters
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
b0 = np.ones(K)*1   #  not sure about this
nu0 = np.ones(K)*2.     # degrees of freedom
"""'The least informative, proper Wishart prior is obtained by setting 
nu = dimensionality', from Wikipedia"""
V_ = np.eye(2)/2  # 'interpreted as a precision matrix'
m_ = np.zeros(2)  # "typically we would choose m0=0 by symmetry"
V0, m0 = [], []
for k in range(K):  # mean and precision needed for each mixture component
    m0.append(m_)
    V0.append(V_)
"a reasonable choice for V would be a prior guess for the precision / n"

alpha0 = a0**2
beta0 = b0**2
W0 = [np.dot(V0[k].T, V0[k]) for k in range(K)]
p_dist_params = {\
                 'alpha': alpha0,
                 'beta': beta0,
                 'm': m0,
                 'W': W0,
                 'nu': nu0,
                 }

# Now intiialising the parameters of q (alpha, beta, m, W)
# chosen arbitrarily ? (can we improve on these?)
a = np.ones(K)
b = np.ones(K)*2
m = [m0[k] + 1. for k in range(K)]
V = [V0[k] + np.eye(2) for k in range(K)]
u = np.ones(K)*1.1


# Sample the model hyperparameters
n = 0 # Taking a single point from the dataset
x = X[n]

# Transforming into parameters, thus imposing constraints
alpha = a**2
beta = b**2
W = [np.dot(V[k].T, V[k]) for k in range(K)]
nu = abs(u) + 2





#%% 3. Sample model parameters, calculate ELBO, calculate gradients, update

step = 0.01 # can't use the same step size for everything - but for now

def multi_sample_ELBO(X,sample_sets,a,b,m,V,u,p_dist_params,K,N):
    # doesn't currently work
    L = 0
    for n in range(N):
        print('Sample no.: ',n)
        Z, pi, mu, lam, r_nk = sample_sets[n]
        with HiddenPrints():
            L = L + calculate_ELBO(X[n],Z,pi,mu,lam,a,b,m,V,u,p_dist_params,K,N)
    return L

def update(x, a, b, m, V, u, p_dist_params, n, step, K, N):
    print('\n***Iteration %d***'%n)
    
    # sample parameters
    alpha = a**2
    beta = b**2
    W = [np.dot(V[k].T, V[k]) for k in range(K)]
    nu = np.abs(u) + 2
    Z, pi, mu, lam, r_nk = samples(alpha, beta, W, m, nu, x, K)
    
    # Calculate gradients and update steps
    if N==1:    # single sample estimate
        L = calculate_ELBO(x,Z,pi,mu,lam,a,b,m,V,u,p_dist_params,K,N)
        # Define gradient functions for each of the updated variables with autograd
        Lgrad_a = grad(calculate_ELBO,5)
        Lgrad_b = grad(calculate_ELBO,6)
        Lgrad_m = grad(calculate_ELBO,7)
        Lgrad_V = grad(calculate_ELBO,8)
        Lgrad_u = grad(calculate_ELBO,9)
        
        with HiddenPrints(): # suppresses print statements in calculate_ELBO
        # Find gradients wrt each variable - use in SGD update
            d_a = Lgrad_a(x,Z,pi,mu,lam,a,b,m,V,u,p_dist_params,K,N)
            d_b = Lgrad_b(x,Z,pi,mu,lam,a,b,m,V,u,p_dist_params,K,N)
            d_m = Lgrad_m(x,Z,pi,mu,lam,a,b,m,V,u,p_dist_params,K,N)
            d_V = Lgrad_V(x,Z,pi,mu,lam,a,b,m,V,u,p_dist_params,K,N)
            d_u = Lgrad_u(x,Z,pi,mu,lam,a,b,m,V,u,p_dist_params,K,N)
            
    else:    # batch/minibatch estimate - doesn't currently work
        sample_sets = []
        for i in range(N):
            sample_set = samples(alpha, beta, W, m, nu, X[n], K)
            sample_sets.append(sample_set)
            
        L = multi_sample_ELBO(X,sample_sets,a,b,m,V,u,p_dist_params,K,N)
        Lgrad_a = grad(multi_sample_ELBO,2)
        Lgrad_b = grad(multi_sample_ELBO,3)
        Lgrad_m = grad(multi_sample_ELBO,4)
        Lgrad_V = grad(multi_sample_ELBO,5)
        Lgrad_u = grad(multi_sample_ELBO,6)
        
        with HiddenPrints(): # suppresses print statements in calculate_ELBO
        # Find gradients wrt each variable - use in SGD update
            d_a = Lgrad_a(X,sample_sets,a,b,m,V,nu,p_dist_params,K,N)
            d_b = Lgrad_b(X,sample_sets,a,b,m,V,nu,p_dist_params,K,N)
            d_m = Lgrad_m(X,sample_sets,a,b,m,V,nu,p_dist_params,K,N)
            d_V = Lgrad_V(X,sample_sets,a,b,m,V,nu,p_dist_params,K,N)
            d_u = Lgrad_u(X,sample_sets,a,b,m,V,nu,p_dist_params,K,N)
            
    # Display updates
    print("\nELBO: = %.3f" % L)
    if n%1 == 0:
        title = 'Update %d' % n
        plot_GMM(X, mu, lam, pi, centres, covs, K, title) 
    print("\n\n***GRADIENTS***\ndL/d_a = ", d_a)
    print("\ndL/d_b = ", d_b)
    print("\ndL/d_m = ", d_m)
    print("\ndL/d_V = ", d_V)
    print('\ndL/du = ', d_u)
    
    # Update alpha, W, beta, m, nu
    step_V, step_a, step_b, step_m, step_u = step, step, step, step, step
    for i in range(len(V)):
        V[i] = V[i] + step_V*d_V[i]
        a[i] = a[i] + step_a*d_a[i]
        b[i] = b[i] + step_b*d_b[i]
        m[i] = m[i] + step_m*d_m[i]
        u[i] = u[i] + step_u*d_u[i]
    return a, b, V, m, u, L

# collect parameters for printing
ELBO = np.zeros(n_its)
betas = np.zeros((K,n_its))
ms = np.zeros((2,K,n_its))
nus = np.zeros((K,n_its))
for j in range(n_its):
    a, b, V, m, u, ELBO[j] = update(X[j], a, b, m, V, u,
                                        p_dist_params,
                                        j, step, K, N)
    betas[:,j] = np.array(b**2).T
    nus[:,j] = np.array(abs(u)+2).T
    ms[:,:,j] = np.array(m).T


plt.figure()
plt.plot(ELBO)
plt.xlabel('Iterations')
plt.title('ELBO')

plt.figure()
plt.plot(betas.T)
plt.title('Betas')

plt.figure()
plt.plot(nus.T)
plt.title('Nus')

plt.figure()
plt.plot(ms[0,:,:],ms[1,:,:])
plt.title('movement of m, mean parameter of NW')

