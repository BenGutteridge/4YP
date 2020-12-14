# -*- coding: utf-8 -*-
"""
Utility functions for GMM experiment

Created on Fri Dec  4 13:37:16 2020

@author: benpg
"""
import autograd.numpy as np
from autograd.numpy import sin, cos
from autograd.scipy.special import gamma, multigammaln
from scipy.stats import wishart # need to replace with own version
from autograd.numpy.random import multivariate_normal
from autograd.scipy.stats import multivariate_normal as norm
from autograd.numpy.linalg import inv, det
import matplotlib.pyplot as plt
from autograd import grad


#%%  Variational distribution q() components - EVALUATION

def q_pi(pi, alpha):
    # output = dirichlet.pdf(pi, alpha)
    output = f_dirichlet(pi,alpha)
    return output

# q(Lambda|nu, W)
def q_lambda(lam, nu, W):
    # Wishart distribution:
    # "The Wishart distribution arises as the distribution of the sample 
    # covariance matrix for a sample from a multivariate normal distribution."
    return f_wishart(lam, W, nu)

# q(mu|beta, m, Lambda)
def q_mu(mu, m, beta, lam):
    return norm.pdf(mu, m, inv(beta*lam))

# PROD_x[PROD_n[r_nk^{z_nk}]]
def q_Z(Z,x,pi,mu,lam):
    qZ = 1
    N = 1   # single sample
    K = pi.shape[0]
    for n in range(N):
        for k in range(K):
            qZ = qZ * responsibility(x,k,pi,mu,lam)**Z[k]
    return qZ



#%%  Variational distribution q() components - SAMPLING

# Dirichlet distribution with K-dimensional input alpha
def sample_pi(alpha):
    return np.random.dirichlet(alpha)

# Wishart distribution for GMM component precisions
def sample_lambda(W, nu):
    return wishart.rvs(df=nu, scale=W)

# Gaussian with ancestral sampling from precicion (from Wishart)
def sample_mu(m, beta, lam):
    return multivariate_normal(m, inv(beta*lam))

# Sample Z = zk with probability = r_nk, the responsibility
def sample_Z(r_nk):
    # r_nk is p(z_nk=1|x), i.e. posterior, so r_nk sums to 1 over K
    # hence multinomial
    return np.random.multinomial(1, r_nk)      

def samples(alpha, beta, W, m, nu, x, K):
    mu, lam = [],[]
    pi = sample_pi(alpha)
    try:
        for k in range(K):
            lam.append(sample_lambda(W[k], nu))
            mu.append(sample_mu(m[k], beta[k], lam[k]))
    except:
        raise Exception('\n\n***ERROR***\n\nEither p or q has gone to zero, causing Elogp or Elogq to go to -Inf\n')
        
    
    # Need to calculate responsibility of each mixture component to sample Z
    r_nk = np.empty(K)
    for k in range(K):
        r_nk[k] = responsibility(x,k,pi,mu,lam)
    Z = sample_Z(r_nk)
    return Z, pi, mu, lam, r_nk


#%% p() component - EVALUATION

# p(X|Z,mu,lambda) -- p(point belongs to Gauss denoted by zk)
def p_X_given_Z_mu_lambda(x,Z,mu,lam):
    # evaluate likelihood wrt partiular mixture component
    k = np.argmax(Z)
    return norm.pdf(x,mu[k],lam[k])

# p(Z|pi) = PROD_k[ pi^znk ]
def p_Z_given_pi(Z, pi):
    # likelihood of random new x being from Gaussian denoted by one-hot Z
    return np.sum(np.multiply(pi,Z))

# as far as I can tell these are all the same as the corresponding q()
def p_pi(pi, alpha):
    return q_pi(pi, alpha)
def p_lambda(lam, nu, W):
    return q_lambda(lam, nu, W)
def p_mu_given_lambda(mu, m, beta, lam):
    return q_mu(mu, m, beta, lam)


#%% Distribution functions

# Dirichlet
def f_dirichlet(pi,alpha):
    B = np.prod(gamma(alpha))/gamma(np.sum(alpha))
    return (1/B)*np.prod(pi**(alpha-1.))

# Wishart
def f_wishart(lam, W, nu):
    d = 2   # dimensionality
    num = (det(lam)**((nu-d-1)/2))*np.exp(-np.trace(np.dot(inv(W),lam))/2)
    den = (2**(nu*d/2))*(det(W)**(nu/2))*multigammaln(nu/2,d)
    return num/den

#%% Major steps

def calculate_ELBO(x, Z, 
                   pi, mu, lam,
                   a, beta, m, V, nu,
                   p_dist_params,
                   K, N):
    # pi, mu, lam = model_params['pi'], model_params['mu'], model_params['lam']
    
    alpha0, beta0, W0, m0, nu0= \
    p_dist_params['alpha'], p_dist_params['beta'], p_dist_params['W'],\
        p_dist_params['m'], p_dist_params['nu'],
    
    alpha = a**2
    W = [np.dot(V[k].T, V[k]) for k in range(K)]

    # 4.1. log p(X,Z,pi,mu,lambda) = 
    #   log[ p(X|Z,mu,lambda).p(Z|pi).p(pi).p(mu|lambda).p(lambda) ]

    Elogp = p_X_given_Z_mu_lambda(x,Z,mu,lam)*\
                p_Z_given_pi(Z, pi)*\
                    p_pi(pi, alpha0)
    for k in range(K):
        Elogp = Elogp * p_mu_given_lambda(mu[k], m0[k], beta0[k], lam[k])*\
                p_lambda(lam[k], nu0, W0[k])
    if np.abs(np.log(Elogp))>1e10 or not np.log(Elogp)==np.log(Elogp):
        print('Elogp: ', Elogp)
    print('\nE[p]: ', Elogp)
    Elogp = np.log(Elogp)
    print('Elogp: ', Elogp)
    
    # 4.2. log q(Z,pi,mu,lambda) = 
    #   log[ q(Z).q(pi).PROD_K[ q(mu|lambda).p(lambda) ] ]
        
    Elogq = q_Z(Z,x,pi,mu,lam)*q_pi(pi,alpha)
    for k in range(K):
        Elogq = Elogq * q_mu(mu[k], m[k], beta[k], lam[k])*\
                q_lambda(lam[k], nu, W[k])
    print('E[q]: ', Elogq)
    Elogq = np.log(Elogq)
    print('Elogq: ', Elogq)
    
    
    L = Elogp - Elogq
    # print("Loss (ELBO) = %.3f" % L)
    return L


#%% Misc

# r_nk = rho_nk/SUM_j(rho_nj)
# where rho_nj = pi_j * N(x|mu_j, (1/Lambda_j))
def responsibility(x, k, pi, mu, lam):
    K = len(pi)
    num = pi[k]*norm.pdf(x,mu[k],inv(lam[k]))
    den = 0
    for j in range(K):
        den = den + pi[j]*norm.pdf(x,mu[j],inv(lam[j]))
    return num/den

# BXL nice plot colours
cols = [\
        '#8159a4',
        '#60c4bf',
        '#f19c39',
        '#cb5763',
        '#6e8dd7',
        ]

# Returns plottable coordinates for ellipses over 2D Gaussians
def draw_ellipse(mu,cov):
    # as I understand it - 
        # diagonalise the cov matrix
        # use eigenvalues for radii
        # use eigenvector rotation from x axis for rotation
    x=mu[0]       #x-position of the center
    y=mu[1]      #y-position of the center
    lam, V = np.linalg.eig(cov) # eigenvalues and vectors
    t_rot = np.arctan(V[1,0]/V[0,0])
    a, b = lam[0], lam[1]    
    t = np.linspace(0, 2*np.pi, 100)
    Ell = np.array([a*np.cos(t) , b*np.sin(t)])  
         #u,v removed to keep the same center location
    R_rot = np.array([[cos(t_rot) , -sin(t_rot)],[sin(t_rot) , cos(t_rot)]])  
         #2-D rotation matrix
    
    Ell_rot = np.zeros((2,Ell.shape[1]))
    for i in range(Ell.shape[1]):
        Ell_rot[:,i] = np.dot(R_rot,Ell[:,i])
    return x+Ell_rot[0,:] , y+Ell_rot[1,:]

def plot_GMM(X, mu, lam, pi, centres, covs, K, title, cols=cols):
    plt.figure()
    plt.plot(X[:,0], X[:,1], 'kx', alpha=0.2)
    
    legend = ['Datapoints']
    
    for k in range(K):
        x_ell, y_ell = draw_ellipse(mu[k], lam[k])
        plt.plot(x_ell, y_ell, cols[k], alpha=pi[k])
        legend.append('k=%d, pi=%.2f'%(k,pi[k]))
        # for whatever reason lots of the ellipses are very long and narrow, why?
        
    # Plotting the ellipses for the GMM that generated the data
    for i in range(2):
        x_true_ell, y_true_ell = draw_ellipse(centres[i], covs[i])
        plt.plot(x_true_ell, y_true_ell, 'g--')
    
    legend.append('Data generation GMM')
    plt.legend(legend)
    plt.title(title)
    plt.show()
    
import os, sys

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout