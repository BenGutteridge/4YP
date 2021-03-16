"""

Main file running VI on a GMM

Ben Gutteridge

"""
# %% Import libraries and own functions/classes

import numpy as np
from numpy.linalg import inv
from copy import deepcopy
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
# Close plots and turn off automatic figure creation (quirk of Spyder)
plt.close('all')
plt.ioff()  

from distribution_classes import JointDistribution, VariationalDistribution
from generate_dataset import generate_2D_dataset
from plot_utils import plot_GMM, E_pi, make_gif, plot_1D_phi as plot_1D_param

# %% Misc setup

"Iterations and update_type"
K = 3                   # Initial number of mixture components
N_its = 10              # Number of iterations of chosen update method performed

# update_type = 'GD'      # Using (true) gradient descent
update_type = 'CAVI'  # Using co-rdinate ascent variational inference algo

"Define parameters of joint distribution (effectively priors on variational params)"
alpha_0 = 1e-3      # Dirichlet prior p(pi) = Dir(pi|alpha0)
m_0 = np.zeros(2)   # Gaussian prior p(mu) = N(mu|m0, C0)
C_0 = np.eye(2)     # Covariance of prior mu
sigma = inv_sigma = np.eye(2)   # Fixed covariance/precision of Gaussian components 
K_inv_sigma = [inv_sigma for _ in range(K)] # for plotting

"Generating dataset"
N = 100
num_clusters = 2
X, centres, covs, weights = generate_2D_dataset(N, K=num_clusters, 
                                       # weights=np.random.dirichlet(np.ones(num_clusters)),
                                       weights = np.ones(num_clusters)/num_clusters)

"Schedule for step sizes in GD (constant at the moment)"
def gd_schedule(step_sizes={'alpha': 1.0, 'm': 1e-2, 'invC': 1e-4}):
    # TODO: fill in with a proper schedule
    return step_sizes

"Initialising params and instantiating distributions"
# TODO: could we move this into VariationalDistribution? Should we avoid dependence on N?
initial_responsibilities = np.random.dirichlet(np.ones(K), N)  # Size (N, K)

variational = VariationalDistribution(initial_responsibilities, 
                                      inv_sigma, update_type, gd_schedule)
joint = JointDistribution(alpha_0, m_0, covariance=C_0)
variational.initialise_params()

"For plotting"
variational_memory = []
filedir = 'plots'
gifdir = 'gifs'

# %% Implement update steps

"Repeat Maximisation and Expectation steps for N_its iterations, maximising ELBO"
for i in tqdm(range(N_its)):
    # Save copy of class with all current params for plotting etc
    variational_memory.append(deepcopy(variational))
    
    # EM steps, calculate ELBO
    variational.M_step(X, joint)
    variational.E_step(X)
    variational.calculate_ELBO(joint)
    
    # Plotting stuff
    title = '(Class) %s: Iteration %d -- ELBO = %7.0f'%(variational.update_type, i, variational.ELBO)
    filename = 'plots/img%04d.png'%i
    variational.calculate_mixing_coefficients()
    plot_GMM(X, variational.means, K_inv_sigma, variational.mixing_coefficients, centres, covs, K, title, savefigpath=filename)
    
# %% Saving animation, plotting ELBO/other key params

# Make and display gif 
gifname = make_gif(filedir, gifdir)
# Empty plots directory for next run
for file in os.listdir(filedir):
  os.remove(os.path.join(filedir,file))
  
# xtracting from memory and plotting salient params
plt.close('all')
ELBOs = np.array([variational_memory[n].ELBO for n in range(1,N_its)])
alphas = np.array([variational_memory[n].alpha for n in range(N_its)])
mixing_coefficients = np.array([variational_memory[n].mixing_coefficients for n in range(1,N_its)])

plot_1D_param(ELBOs.reshape(-1,1), 'ELBO', 1)
plot_1D_param(alphas, 'alphas', K)
plot_1D_param(mixing_coefficients, 'Mixing coefficients (E[pi_k])', K)
plt.show()
