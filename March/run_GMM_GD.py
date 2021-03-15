# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 11:18:46 2021

@author: benpg
"""

# General libraries
import numpy as np
from numpy.linalg import inv
from tqdm import tqdm

# Own functions
from generate_dataset import generate_2D_dataset
from calculate_responsibilities import * 
from plot_utils import *
from EM_steps import E_step, M_step, M_step_GD, perturb_variational_params
from calculate_ELBO import calculate_ELBO
plt.close('all')
plt.ioff()

# Dataset and update params
N_its = 100
N = 100
K = 5
D = 2

N_clusters = 2
# centres = [np.array([1,8]), np.array([6,5])]
# covs = [np.eye(2), np.eye(2)]
weights = np.array([0.5,0.5])
X, centres, covs = generate_2D_dataset(N, K=N_clusters, 
                                       # centres=centres, covs=covs,
                                       weights=weights,
                                       )

# Variational priors
alpha0 = 1e-3     # as alpha0 -> 0, pi_k -> 0. As alpha0 -> Inf, pi_k -> 1/K
m0 = np.zeros(2)  # zero by convention (symmetry)
C0 = invC0 = np.eye(2) # identity by convention
invSig = np.eye(2)# Assuming all clusters have identity covariance
KinvSig = [invSig for _ in range(K)] # for plotting

# Saving plots and making gif
filedir = 'plots'
gifdir = 'gifs'


r = np.array([np.random.dirichlet(np.ones(K)) for _ in range(N)])
r = [r[:,k] for k in range(K)]

# for plotting
alphas = np.zeros((N_its,K))
NKs = np.zeros((N_its,K))
ms = np.zeros((N_its,K,D))
varx, vary, covxy = np.empty((N_its,K)),np.empty((N_its,K)),np.empty((N_its,K)) 

verbose = False

# Intialise variational parameters
alpha, m, C, _, _, _ = M_step(r,X,alpha0,m0,invC0,invSig,K)
alpha, m, C = perturb_variational_params(
                                        alpha=alpha, 
                                        m=m, 
                                        C=C,
                                        non_diag=True,
                                        )

ELBO, ELBO_M, ELBO_E = np.empty(2*N_its), np.empty(N_its), np.empty(N_its)
for i in tqdm(range(N_its)):
    # Plot
    Epi = E_pi(alpha, alpha0, N)
    title = 'GD: iteration %d' % i
    filename = 'plots/img%04d.png'%i
    plot_GMM(X, m, KinvSig, Epi, centres, covs, K, title, savefigpath=filename)
      
    # M step
    alpha, m, C, NK, xbar, SK = M_step_GD(r, X, alpha, m, C, alpha0, m0, invC0, invSig, K,
                step_sizes={'alpha': 1.0, 'm': 1e-2, 'invC': 1e-5},
                )
    ELBO[2*i] = calculate_ELBO(r, alpha, m, C, invSig, alpha0, m0, C0, NK, xbar, SK)
    ELBO_M[i] = ELBO[2*i]
    
       
    # E step
    r = E_step(N, K, alpha, m, C, invSig, X)
    ELBO[2*i+1] = calculate_ELBO(r, alpha, m, C, invSig, alpha0, m0, C0, NK, xbar, SK)
    ELBO_E[i] = ELBO[2*i+1]
    
    # Plot stuff
    alphas[i,:] = alpha
    NKs[i,:] = NK
    # m is shape K,D when cast to nparray
    # ms[i,:,:] = np.array(m) # shape N_its*K*D
    varx[i,:] = np.array([C[k][0,0] for k in range(K)]) 
    vary[i,:] = np.array([C[k][1,1] for k in range(K)]) 
    covxy[i,:] = np.array([C[k][1,0] for k in range(K)]) 
  
    if verbose:
        print('\n******************Iteration %d************************\n'%i)
        print('alpha', alpha, '\nm', m, '\nC', C)
        print('E[pi] = ', Epi)
        # print('ELBO = %f'%ELBO[i])
  
# Make and display gif 
gifname = make_gif(filedir, gifdir)
# delete pngs for next run
for file in os.listdir(filedir):
  os.remove(os.path.join(filedir,file))
  
  
# Plot parameters
plt.close('all')
plot_ELBO(ELBO, ELBO_E, ELBO_M, N_its)
plot_1D_phi(alphas, 'alphas', K)
plot_1D_phi(NKs, 'Nk', K)
plot_K_covs(varx, vary, covxy, K)
plt.show()