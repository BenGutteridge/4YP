# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 18:51:00 2021

@author: benpg

Optimise mean only, using true gradient ascent on a GMM where all other
variational params are pre-trained by CAVI 
"""

from EM_steps_unknown_cov import E_step, M_step, M_step_GD
from calculate_ELBO_unknown_cov import calculate_ELBO
from generate_dataset import generate_2D_dataset
from plot_utils import plot_GMM, E_pi, make_gif, plot_ELBO
from phi_grad_funcs_unknown_cov import Dm as grad_m
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
plt.ioff()
plt.close('all')

# Dataset and update params
N_its = 100
step = 1e-4
N_its_pretrain = 100
K = 4
D = 2

N = 300     # number of datapoints
N_clusters = 3
X, centres, covs = generate_2D_dataset(N, K=N_clusters)

# Variational priors
alpha0 = 1e-3     # as alpha0 -> 0, pi_k -> 0. As alpha0 -> Inf, pi_k -> 1/K
beta0 = 1e-10     # ???
m0 = np.zeros(2)  # zero by convention (symmetry)
W0 = np.eye(2)    # 
nu0 = 2           # 

# Saving plots and making gif
verbose = True
filedir = 'plots'
gifdir = 'gifs'

r = np.array([np.random.dirichlet(np.ones(K)) for _ in range(N)])
r = [r[:,k] for k in range(K)]

# Pretrain model
print('\n***PRETRAINING MODEL w/ CAVI (unknown covariance)***\n',
      flush=True) # flush stops weird tqdm bug
for _ in tqdm(range(N_its_pretrain)):
    alpha, beta, m, W, nu, NK, xbar, S = M_step(r,X,alpha0,beta0,m0,W0,nu0,K)
    r = E_step(N,K,alpha,nu,W,beta,m,X)

# %% Shifting pretrained model and retraining using true GD (only m) 

# Shifting alpha as well
alpha = np.ones(K) * (N/K)

# Shift m (variational distribution of mean)
for k in range(K):
  m[k] += np.random.rand(2,)*5
Epi = E_pi(alpha, alpha0, N)
plot_GMM(X, m, inv(S), Epi, centres, covs, K, title='After CAVI, shifting m')
plt.title('Plots after training w/ CAVI and shifting m')
plt.show()

# Retrain model
ELBO, ELBO_M, ELBO_E = np.empty(2*N_its), np.empty(N_its), np.empty(N_its)
print('\n***TRAINING w/ GD over variational parameter m***\n',
      flush=True) # flush stops weird tqdm bug
for i in tqdm(range(N_its)):
    # M step (using GD to update m)
    alpha, beta, m, W, nu, NK, xbar, S = M_step_GD(X, r, alpha, beta, m, W, nu, alpha0, beta0, m0, W0, nu0, K, D, step=step)
    ELBO[2*i] = calculate_ELBO(r,alpha,beta,m,W,nu,NK,S,xbar,alpha0,beta0,m0,W0,nu0)
    ELBO_M[i] = ELBO[2*i]

    # E step
    r = E_step(N,K,alpha,nu,W,beta,m,X)
    ELBO[2*i+1] = calculate_ELBO(r,alpha,beta,m,W,nu,NK,S,xbar,alpha0,beta0,m0,W0,nu0)
    ELBO_E[i] = ELBO[2*i+1]
  
    if verbose:
        print('\n******************Iteration %d************************\n'%i)
        print('\nalpha', alpha, '\nbeta', beta, '\nm', m, '\nW', W, '\nnu', nu)
        print('E[pi] = ', Epi)
        print('ELBO = %f'%ELBO[i])

    # Plot
    Epi = E_pi(alpha, alpha0, N)
    title = 'GD for m: iteration %d' % i
    filename = 'plots/img%04d.png'%i
    # plot_GMM(X, mu, lam, pi, centres, covs, K, title)
    plot_GMM(X, m, inv(S), Epi, centres, covs, K, title, savefigpath=filename)


plot_ELBO(ELBO, ELBO_E, ELBO_M, N_its)
plt.show()

# Make and display gif 
gifname = make_gif(filedir, gifdir)
# delete pngs for next run
for file in os.listdir(filedir):
  os.remove(os.path.join(filedir,file))
