import numpy as np
from numpy.linalg import inv
from distribution_classes import JointDistribution, VariationalDistribution
from generate_dataset import generate_2D_dataset

from plot_utils import plot_GMM, E_pi, make_gif, plot_1D_phi as plot_1D_param

from EM_steps import perturb_variational_params
from copy import deepcopy
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

plt.close('all')
plt.ioff()
            

# "priors"
alpha_0 = 1e-3
m_0 = np.zeros(2)
C_0 = np.eye(2)
K = 5

# Iterations and runtype
N_its = 40
runtype = 'GD (perturbed M-step init)'
# runtype = 'GD (random init)'
# runtype = 'CAVI'

# Generate dataset
N = 100
num_clusters = 2
X, centres, covs = generate_2D_dataset(N, K=num_clusters, 
                                       weights=np.array([0.5,0.5]),
                                       )

# Fixed covariance of every component of the Gaussian mixture
sigma = inv_sigma = np.eye(2)
K_inv_sigma = [inv_sigma for _ in range(K)]

initial_responsibilities = np.random.dirichlet(np.ones(K), N)  # Size (N, K)

variational = VariationalDistribution(initial_responsibilities, inv_sigma, runtype)
joint = JointDistribution(alpha_0, m_0, covariance=C_0)

# TODO: move to separate file
# def calculate_ELBO(variational, joint):
#     """Based on equation 10.70-10.77 from Bishop, with some modifications due to our fixed covariance."""
#     alpha0 = joint.alpha
#     m0 = joint.mean
#     C0 = joint.covariance
#     alpha = variational.alpha
#     m = variational.means
#     C = variational.covariances
#     r = variational.responsibilities
#     NK = variational.NK
#     xbar = variational.xbar
#     SK = variational.SK
#     invSig = variational.inv_sigma

#     p1 = E_ln_p_X_given_Z_mu(m, invSig, C0, NK, xbar, SK, D=2)  # ~ Eqn 10.71
#     p2 = E_ln_p_Z_given_pi(r, alpha)                            # = Eqn 10.72
#     p3 = E_ln_p_pi(alpha0, alpha)                               # = Eqn 10.73
#     p4 = E_ln_p_mu(m, C, m0, inv(C0), D=2)                      # ~ Eqn 10.74
#     q1 = E_ln_q_Z(r)                                            # = Eqn 10.75
#     q2 = E_ln_q_pi(alpha)                                       # = Eqn 10.76
#     q3 = E_ln_q_mu(C, D=2)                                      # ~ Eqn 10.77
#     return p1+p2+p3+p4-q1-q2-q3

variational_memory = []
ELBO = np.zeros(N_its)
# Saving plots and making gif
filedir = 'plots'
gifdir = 'gifs'

#TODO: have an intiialise params method in variational, and move these into it
if runtype == 'GD (perturbed M-step init)':
    variational.M_step_CAVI(X, joint)
    variational.alpha, variational.means, variational.covariances = perturb_variational_params(alpha=variational.alpha, 
                                                                                                m=variational.means, 
                                                                                                C=variational.covariances, 
                                                                                                non_diag=True)
    variational.update_precision_from_covariance() # need to make sure both change whenever one does
    
if runtype == 'GD (random init)':
    variational.alpha = np.ones(K)*10#(X.shape[0]/K) # this is the same as the above
    variational.means = 2*np.random.rand(K,2) - 1
    variational.covariances = variational.precisions = np.array([np.eye(2) for _ in range(K)]) # same as above, diag

for i in tqdm(range(N_its)):
    variational_memory.append(deepcopy(variational))
    variational.M_step(X, joint)
    variational.E_step(X)
    title = '(Class) %s: Iteration %d'%(variational_memory[-1].update_type, i)
    filename = 'plots/img%04d.png'%i
    Epi = E_pi(variational.alpha, joint.alpha, X.shape[0])
    plot_GMM(X, variational_memory[-1].means, K_inv_sigma, Epi, centres, covs, K, title,  savefigpath=filename)
    variational.calculate_ELBO(joint)
    
# Make and display gif 
gifname = make_gif(filedir, gifdir)
# delete pngs for next run
for file in os.listdir(filedir):
  os.remove(os.path.join(filedir,file))
  
plt.close('all')
plt.figure()
ELBOs = np.array([variational_memory[n].ELBO for n in range(1,N_its)])
alphas = np.array([variational_memory[n].alpha for n in range(N_its)])
plt.plot(ELBOs)
plt.title('ELBO')
plt.xlabel('Iterations')
plot_1D_param(ELBOs.reshape(-1,1), 'ELBO', 1)
plot_1D_param(alphas, 'alphas', K)
plt.show()
