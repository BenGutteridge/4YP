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
from plot_utils import plot_GMM, E_pi, make_gif, plot_1D_phi as plot_1D_param, plot_K_covs, plot_cov_ellipses_gif

np.random.seed(41)

# %% Misc setup

"Iterations and update_type"
K = 6                   # Initial number of mixture components
N_its = 100             # Number of iterations of chosen update method performed

# update_type = 'GD'      # Using (true) gradient descent
# update_type = 'CAVI'  # Using co-rdinate ascent variational inference algo
update_type = 'SGD'
# update_type = 'SNGD'
print('\n%s' % update_type)
minibatch_size = 10

"Define parameters of joint distribution (effectively priors on variational params)"
alpha_0 = 1e-3      # Dirichlet prior p(pi) = Dir(pi|alpha0)
m_0 = np.zeros(2)   # Gaussian prior p(mu) = N(mu|m0, C0)
C_0 = np.eye(2)     # Covariance of prior mu
inv_sigma = np.eye(2)   # Fixed covariance/precision of Gaussian components 
K_inv_sigma = [inv_sigma for _ in range(K)] # for plotting

"Generating dataset"
N = 500
num_clusters = 4
X, centres, covs, weights = generate_2D_dataset(N, K=num_clusters,
                                       # weights=np.random.dirichlet(np.ones(num_clusters)),
                                       weights = np.ones(num_clusters)/num_clusters)

# # trying with a real dataset
# X = np.loadtxt(r"C:\Users\benpg\Documents\4YP\Datasets\s1.txt")
# X = X/1e5
# N = X.shape[0]
# K = 20
# K_inv_sigma = [inv_sigma for _ in range(K)]
# centres, covs = None, None

"""Schedule for step sizes in GD. Constant by default (no t input) or a decaying
step size. forgetting rate is between 0.5 and 1 and indicates how quickly old
info is forgotten, delay >= 0 and downweights early iterations."""
def gd_schedule(t=None, scale=1, delay=1., forgetting=0.5, 
                step_sizes={'alpha': 1.0, 'm': 1e-2, 'C': 1e-3, 
                            'lam1': 1e-4, 'lam2': 1e-4}): # maybe use kwargs?
    if t is not None:
        # rho_t = scale*(t + delay)**(-forgetting) # Eq 26, Hoffman SVI
        rho_t = scale*np.exp(-0.05*t)   # off the top of my head
        # decay, A = 1, 0               # A is stability constant >= 0
        # rho_t = scale/(t+1+A)**decay  # See Spall 4.14
        steps= {}
        for key in step_sizes:
            if update_type == 'SGD':
                steps[key] = step_sizes[key] * rho_t 
            elif update_type == 'SNGD':
                steps[key] = 1*rho_t
    return steps

"Initialising params and instantiating distributions"
# TODO: could we move this into VariationalDistribution? Should we avoid dependence on N?

initial_responsibilities = np.random.dirichlet(np.ones(K), N)

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
    samples = np.random.choice(N, size=minibatch_size, replace=False)
 
    # # EM steps, calculate ELBO
    variational.E_step(X, samples)
    variational.M_step(X, joint, samples, t=i)
    variational.calculate_weighted_statistics(X)
    variational.calculate_ELBO(joint)
    
    # Plotting stuff
    title = '%s: Iteration %d -- ELBO = %7.0f'%(variational.update_type, i, variational.ELBO)
    if update_type == 'SGD' or update_type == 'SNGD':
        title = 'Minibatch: %d, ' % minibatch_size + title
    filename = 'plots/img%04d.png'%i
    variational.calculate_mixing_coefficients()
    plot_GMM(X, variational.means, K_inv_sigma, variational.mixing_coefficients, centres, covs, K, title, savefigpath=filename, xylims=None)
    
# %% Saving animation, plotting ELBO/other key params

"Make and display gif of mixture model over time" 
gifname = make_gif(filedir, gifdir, gifname=update_type)
# Empty plots directory for next run
for file in os.listdir(filedir):
  os.remove(os.path.join(filedir,file))
  
"extracting from memory, and plotting, salient params"
plt.close('all')
ELBOs = np.array([variational_memory[n].ELBO for n in range(1,N_its)])
alphas = np.array([variational_memory[n].alpha for n in range(N_its)])
mixing_coefficients = np.array([variational_memory[n].mixing_coefficients for n in range(1,N_its)])
varx = np.array([variational_memory[n].covariances[:,0,0] for n in range(N_its)])
vary = np.array([variational_memory[n].covariances[:,1,1] for n in range(N_its)])
covxy = np.array([variational_memory[n].covariances[:,1,0] for n in range(N_its)])

plot_1D_param(ELBOs.reshape(-1,1), 'ELBO', 1)
plot_1D_param(alphas, 'alphas', K)
# plot_1D_param(mixing_coefficients, 'Mixing coefficients (E[pi_k])', K)
# plot_K_covs(varx,vary,covxy,K)
plt.show()

# if update_type == 'GD':
#     d_means_mag = np.array([variational_memory[n].means for n in range(1,N_its)])

"""Makes animation showing the ellipses for the distribution of 
\mu_k ~ N{\mu_k|m_k, C_k} over time"""
# plot_cov_ellipses_gif(variational_memory, N_its, K, gifname=' --- Mu distribution')

"""plot base step size"""
rho_t = (gd_schedule(t=np.arange(N_its))['alpha'])
plt.figure()
plt.plot(rho_t)
plt.xlabel('Iterations')
plt.title('rho_t, base step size')
plt.show()
