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
            qZ *= responsibility(x,k,pi,mu,lam)**Z[k]
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
    return multivariate_normal(m, (beta*lam)**(-1))

# Sample Z = zk with probability = r_nk, the responsibility
def sample_Z(r_nk):
    # r_nk is p(z_nk=1|x), i.e. posterior, so r_nk sums to 1 over K
    # hence multinomial
    return np.random.multinomial(1, r_nk)      


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


#%% Misc

# r_nk = rho_nk/SUM_j(rho_nj)
# where rho_nj = pi_j * N(x|mu_j, (1/Lambda_j))
def responsibility(x, k, pi, mu, lam):
    K = len(pi)
    num = pi[k]*norm.pdf(x,mu[k],inv(lam[k]))
    den = 0
    for j in range(K):
        den += pi[j]*norm.pdf(x,mu[j],inv(lam[j]))
    return num/den

# BXL nice plot colours
cols = [\
        '#8159a4',
        '#60c4bf',
        '#f19c39',
        '#cb5763',
        '#6e8dd7',
        ]

# Plots ellipses over 2D Gaussians
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