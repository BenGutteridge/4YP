# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 17:54:19 2020

@author: benpg
"""
# External
import autograd.numpy as np
from autograd.numpy.random import multivariate_normal
from autograd import grad
import matplotlib.pyplot as plt
plt.close('all')
from numpy.linalg import inv, cholesky

# Misc
from utils import samples, draw_ellipse, plot_GMM, calculate_ELBO, HiddenPrints, cols

import pickle

# playing with samples

# Define GMMs for generation
centres = [np.array([0.,3.]), np.array([2.,0.])]
covs = [np.eye(2), np.array([[0.6,0.4],
                             [0.4,0.6]])]

K = 2

N_total = 1000  # total number of datapoints wanted
X1 = multivariate_normal(mean=centres[0],
                         cov=covs[0],
                         size=int(N_total/2))
X2 = multivariate_normal(mean=centres[1],
                         cov=covs[1],
                         size=int(N_total/2))
X = np.concatenate((X1,X2))

a = np.ones(2)*(10**0.5)  # large alpha means pi values are ~=
b = np.ones(2)*(1000**0.5)  # large beta keeps Gaussian from which mu is drawn small
V = [inv(cholesky(covs[k]))/(1000**0.5) for k in range(K)]
m = centres
u = np.ones(2)*(1000) - 2

alpha, beta, nu = a**2, b**2, abs(u)+2
W = [np.dot(V[k].T,V[k]) for k in range(K)]

# alpha = np.ones(2)*10  # large alpha means pi values are ~=
# beta = np.ones(2)*1000  # large beta keeps Gaussian from which mu is drawn small
# W = [inv(covs[k])/1000 for k in range(K)]
# m = centres
# nu = np.ones(2)*1000

informative_priors = [a,b,V,m,u]
with open('informative_priors2.pkl', 'wb') as f:
    pickle.dump(informative_priors, f)
"""
nu = d is the least informative prior, and the smallest, so  a large nu
is more informative. On its own this makes the lam samples huge, so
we also give W a small prior, the scale - intuitively saying that we are 
sampling lam from a small space.

"the prior mean of Wishart(W,nu) is nu*W, so a reasonable choice for W would 
be a prior guess for the precision / nu"
This is what we have done here, divided W by nu. Nice!
"""

# the trick to informative priors - 
# alpha doesnt much matter, apparently


title = ''
for i in range(10):
    title = '%d'%i
    Z, pi, mu, lam, r_nk = samples(alpha, beta, W, m, nu, X[0], K) 
    plot_GMM(X, mu, lam, pi, centres, covs, K, title, cols=['r', 'b'])

    inv_lam_beta = [inv(beta[k]*lam[k]) for k in range(K)]
    ccols = ['r--', 'b--']
    for k in range(K):
        x,y = draw_ellipse(m[k],inv_lam_beta[k])
        plt.plot(x,y,ccols[k])