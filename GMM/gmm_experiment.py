# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 12:14:06 2020

@author: benpg
"""

# import torch
# from torch import tensor

import autograd.numpy as np
from autograd.numpy.random import multivariate_normal
from autograd.numpy.linalg import inv
from numpy import sin, cos
from autograd import grad
from scipy.stats import dirichlet, wishart
from autograd.scipy.stats import multivariate_normal as norm
from autograd.scipy.special import gamma

import matplotlib.pyplot as plt
plt.close('all')

# Single sample
N = 1
# How many mixture components do we want to start with
K = 5

# 1. Create a dataset and plot it
# must be 2D
centres = [np.array([0.,3.]), np.array([2.,0.])]
covs = [np.eye(2), np.array([[0.6,0.4],
                             [0.4,0.6]])]
X1 = multivariate_normal(mean=centres[0],
                         cov=covs[0],
                         size=(500))
X2 = multivariate_normal(mean=centres[1],
                         cov=covs[1],
                         size=(500))
X = np.concatenate((X1,X2))
plt.plot(X[:,0], X[:,1], 'kx')
# GMM has true distribution of 0.5(N(30,5) + N(70,3))


# 2. Define equations

def f_dirichlet(pi,alpha):
    B = np.prod(gamma(alpha))/gamma(np.sum(alpha))
    return (1/B)*np.prod(pi**(alpha-1.))

def q_pi(pi, alpha):
    # output = dirichlet.pdf(pi, alpha)
    output = f_dirichlet(pi,alpha)
    return output
def sample_pi(alpha):
    return np.random.dirichlet(alpha)

# q(Lambda|nu, W)
def q_lambda(lam, nu, W):
    # Wishart distribution:
    # "The Wishart distribution arises as the distribution of the sample 
    # covariance matrix for a sample from a multivariate normal distribution."
    return wishart.pdf(lam, nu, W)
def sample_lambda(W, nu):
    return wishart.rvs(df=nu, scale=W)

# q(mu|beta, m, Lambda)
def q_mu(mu, m, beta, lam):
    return norm.pdf(mu, m, inv(beta*lam))
def sample_mu(m, beta, lam):
    return multivariate_normal(m, (beta*lam)**(-1))

# r_nk = rho_nk/SUM_j(rho_nj)
# where rho_nj = pi_j * N(x|mu_j, (1/Lambda_j))
def responsibility(x, k, pi, mu, lam):
    K = len(pi)
    num = pi[k]*norm.pdf(x,mu[k],inv(lam[k]))
    den = 0
    for j in range(K):
        den += pi[j]*norm.pdf(x,mu[j],inv(lam[j]))
    return num/den

# PROD_x[PROD_n[r_nk^{z_nk}]]
def q_Z(Z,x,pi,mu,lam):
    qZ = 1
    N = 1   # single sample
    K = pi.shape[0]
    for n in range(N):
        for k in range(K):
            qZ *= responsibility(x,k,pi,mu,lam)**Z[k]
    return qZ

# how do you sample q(Z)??
# come back to
def sample_Z(r_nk):
    # has the same functional /form/ as p(Z|pi), but it's not just pi
    # you sample Z = zk with probability r_nk
    # r_nk is p(z_nk=1|x), i.e. posterior, so r_nk sums to 1 over K
    return np.random.multinomial(1, r_nk)        


# 3. Sample q(Z,pi,mu,lambda) for all of the latent variables

# Initial parameters of prior distributions of q
alpha = np.ones(K) 
# "by symmetry we have chosen the same factor for each of the components"
# "if alpha0 is small ... influenced primarily by data rather than priors"
beta = np.ones(K)   # ?
nu = 2     # degrees of freedom
"""'The least informative, proper Wishart prior is obtained by setting 
nu = dimensionality', from Wikipedia"""
W0 = np.eye(2)/nu  # 'interpreted as a precision matrix'
m0 = np.zeros(2)  # "typically we would choose m0=0 by symmetry"
W, m = [], []
for k in range(K):
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


r_nk = np.empty(K)
for k in range(K):
    r_nk[k] = responsibility(x,k,pi,mu,lam)
Z = sample_Z(r_nk)


print("\nmu = ", mu, 
      "\nlams = ", lam, 
      "\npi = ", pi, 
      '\nr_nk = ', r_nk,
      "\nZ = ", Z)

def draw_ellipse(mu,cov,col=None):
    # as I understand it - 
        # diagonalise the cov matrix
        # use eigenvalues for radii
        # use eigenvector rotation from x axis for rotation
    x=mu[0]       #x-position of the center
    y=mu[1]      #y-position of the center
    lam, V = np.linalg.eig(cov)
    t_rot = np.arctan(V[1,0]/V[0,0])
    a, b = lam[0], lam[1]
    # a=cov[0,0]       #radius on the x-axis
    # b=cov[1,1]      #radius on the y-axis
    # t_rot=cov[1,0] #rotation angle
    
    t = np.linspace(0, 2*np.pi, 100)
    Ell = np.array([a*np.cos(t) , b*np.sin(t)])  
         #u,v removed to keep the same center location
    R_rot = np.array([[cos(t_rot) , -sin(t_rot)],[sin(t_rot) , cos(t_rot)]])  
         #2-D rotation matrix
    
    Ell_rot = np.zeros((2,Ell.shape[1]))
    for i in range(Ell.shape[1]):
        Ell_rot[:,i] = np.dot(R_rot,Ell[:,i])
    # plt.plot(x+Ell_rot[0,:] , y+Ell_rot[1,:], col)
    return x+Ell_rot[0,:] , y+Ell_rot[1,:]
cols = [\
        '#8159a4',
        '#60c4bf',
        '#f19c39',
        '#cb5763',
        '#6e8dd7',
        ]
for k in range(K):
    x_ell, y_ell = draw_ellipse(mu[k], lam[k], cols[k])
    plt.plot(x_ell, y_ell, cols[k])
    # for whatever reason lots of the ellipses are very long and narrow, why?
    
# checking
for i in range(2):
    x_true_ell, y_true_ell = draw_ellipse(centres[i], covs[i])
    plt.plot(x_true_ell, y_true_ell, 'g--')
    
# 4. Plug samples from q into L = E[log p] + E[log q] 

# 4.1. log p(X,Z,pi,mu,lambda) = 
    # log[ p(X|Z,mu,lambda).p(Z|pi).p(pi).p(mu|lambda).p(lambda) ]

# Define functions of p
def p_X_given_Z_mu_lambda(x,Z,mu,lam):
    # evaluate likelihood wrt partiular mixture component
    k = np.argmax(Z)
    return norm.pdf(x,mu[k],lam[k])

def p_Z_given_pi(Z, pi):
    # likelihood of random new x being from Gaussian denoted by one-hot Z
    return np.sum(np.multiply(pi,Z))

# as far as I can tell these are all the same?
def p_pi(pi, alpha):
    return q_pi(pi, alpha)
def p_lambda(lam, nu, W):
    return q_lambda(lam, nu, W)
def p_mu_given_lambda(mu, m, beta, lam):
    return q_mu(mu, m, beta, lam)

def calculate_ELBO(x,Z,pi,mu,lam,alpha,beta,W,m,nu,K,N):
    
    Elogp = p_X_given_Z_mu_lambda(x,Z,mu,lam)*\
                p_Z_given_pi(Z, pi)*\
                    p_pi(pi, alpha)
    for k in range(K):
        Elogp *= p_mu_given_lambda(mu[k], m[k], beta[k], lam[k])*\
                p_lambda(lam[k], nu, W[k])
    Elogp = np.log(Elogp)
    
    
    # 4.2. log q(Z,pi,mu,lambda) = 
        # log[ q(Z).q(pi).PROD_K[ q(mu|lambda).p(lambda) ] ]
        
    Elogq = q_Z(Z,x,pi,mu,lam)*q_pi(pi,alpha)
    for k in range(K):
        Elogq *= q_mu(mu[k], m[k], beta[k], lam[k])*\
                q_lambda(lam[k], nu, W[k])
    Elogq = np.log(Elogq)
    
    L = Elogp - Elogq
    # print("Loss (ELBO) = %.3f" % L)
    return L
# L is -ve, it's a LB on the log evidence, i.e. a log probability, 
# log of something <= 1, so it MUST be -ve.
# All log probbalities are <= 0

# 5. Implement autograd, find gradient of L wrt all parameters:
    # alpha, beta,m and W
    # assume nu remains the same here (need a better understanding of Wishart)

L = calculate_ELBO(x,Z,pi,mu,lam,alpha,beta,W,m,nu,K,N)
print("Loss = %.3f" % L)

Lgrad_alpha = grad(calculate_ELBO,5)
Lgrad_beta = grad(calculate_ELBO,6)
Lgrad_W = grad(calculate_ELBO,7)
Lgrad_m = grad(calculate_ELBO,8)

d_alpha = Lgrad_alpha(x,Z,pi,mu,lam,alpha,beta,W,m,nu,K,N)
d_beta = Lgrad_alpha(x,Z,pi,mu,lam,alpha,beta,W,m,nu,K,N)
d_W = Lgrad_alpha(x,Z,pi,mu,lam,alpha,beta,W,m,nu,K,N)
d_m = Lgrad_alpha(x,Z,pi,mu,lam,alpha,beta,W,m,nu,K,N)

print("dL/d_alpha = ", d_alpha)
print("dL/d_beta = ", d_beta)
print("dL/d_W = ", d_W)
print("dL/d_m = ", d_m)

# these all give the same result - there is a mistake, but the programme 
# doesn't tell us so


"""
To Do
- Once that is all set up, check you can evaluate L
- Then sort out the differential of L, use autograd
- Plot updating model as you update mu, sig, pi
    - q: how do you ever get a converged model if you don't converge mu/pi/sig,
    you'll only be able to sample them???
-Get it to calculate true gradient
- Get plots indicating approx gradient and climbing L function (plot)
    - one plot per hyperparameter
    - turn into gifs w/ that lib, check twitter
    

"""



