import numpy as np
from numpy.linalg import inv
from statistics_of_observed_data import N_k, x_k_bar, S_k
from CAVI_updates import alpha_k, m_invC_k
from generate_dataset import generate_2D_dataset
from calculate_responsibilities import r_nk
from plot_utils import plot_GMM, E_pi, make_gif
from calculate_ELBO import E_ln_p_X_given_Z_mu, E_ln_p_Z_given_pi, E_ln_p_pi, E_ln_p_mu
from calculate_ELBO import E_ln_q_Z, E_ln_q_pi, E_ln_q_mu
from grad_funcs import L_grad_alpha, L_grad_m, L_grad_invC
from EM_steps import perturb_variational_params
from copy import deepcopy
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

plt.close('all')
plt.ioff()

"""TODO: move classes to separate file. Leave this as 'run' function
Also, have some sort of 'gd_schedule' arg - simple at first but can add more complexity
query it at each iteration
"""
class JointDistribution:
    def __init__(self, alpha, mean, covariance=None, precision=None):
        """

        :param alpha: Parameter of Dirichlet distribution for mixture weights
        :param mean:
        :param covariance:
        :param precision:
        """
        self.alpha = alpha
        self.mean = mean

        if covariance is not None:
            self.covariance = covariance
            self.precision = np.linalg.inv(covariance)
        elif precision is not None:
            self.precision = precision
            self.covariance = inv(precision)
        else:
            raise ValueError("Must specify one of precision and covariance.")


class VariationalDistribution:
    """TODO: Briefly document the properties of this class, including the shapes of all arrays

    """
    ALPHA_LB = 0.1

    def __init__(self, initial_responsibilities, inv_sigma, update_type,
                 step_sizes={'alpha': 1.0, 'm': 1e-2, 'invC': 1e-4}):
        self.update_type = update_type
        self.step_sizes = step_sizes

        self.responsibilities = initial_responsibilities

        self.K = initial_responsibilities.shape[1]
        self.D = 2

        self.inv_sigma = inv_sigma
        self.alpha = np.zeros(self.K)
        self.means = np.zeros((self.K, self.D))
        self.precisions = self.covariances = np.zeros((self.K, self.D, self.D))

        self.NK = np.zeros(self.K)
        self.xbar = np.zeros((self.K,self.D))
        self.SK = np.zeros((self.K,self.D,self.D))
        
    def calculate_weighted_statistics(self, X):
        """TODO: Describe what NK, xbar and SK are in docstring. Describe this function's behaviour/purpose."""
        r = self.responsibilities

        for k in range(self.K):
            self.NK[k] = N_k(r[:,k])
            self.xbar[k] = x_k_bar(self.NK[k], r[:,k], X)
            self.SK[k] = S_k(self.NK[k], r[:,k], X, self.xbar[k])

    def M_step(self, X, joint_distribution):
        if self.update_type == "CAVI":
            self._M_step_CAVI(X, joint_distribution)
        elif self.update_type == "GD":
            self._M_step_GD(X, joint_distribution)

    def _M_step_CAVI(self, X, joint_distribution):
        self.calculate_weighted_statistics(X)

        for k in range(self.K):
            # TODO: rename, get rid of _k everywhere
            self.alpha[k] = alpha_k(self.NK[k], joint_distribution.alpha)
            self.means[k], self.precisions[k] = \
                m_invC_k(self.NK[k], self.xbar[k], joint_distribution.mean, joint_distribution.precision, self.inv_sigma)
            self.covariances[k] = inv(self.precisions[k])
            
    def _M_step_GD(self, X, joint_distribution):
        # performing GD update for M step rather than CAVI update equations
        self.calculate_weighted_statistics(X)

        # TODO: consider putting gradient step stuff in two separate methods, one for computing the updates and one for
        # applying the updates

        # Compute gradient updates
        # TODO: point to equation in Bishop or elsewhere whenever you've just reproduced something
        self.d_alpha = L_grad_alpha(joint_distribution.alpha, self.alpha, self.NK)
        self.d_m = L_grad_m(self.means, joint_distribution.mean, joint_distribution.precision, self.inv_sigma, self.NK, self.xbar)
        self.d_invC = L_grad_invC(self.means, self.precisions, self.inv_sigma, joint_distribution.precision, self.NK)

        # Apply gradient updates
        for k in range(self.K): 
            # constraints: alpha>0. Setting alpha>.1 as psi'(alpha->0) -> inf
            self.alpha[k] = np.max((self.ALPHA_LB, self.alpha[k] + (self.d_alpha[k]*self.step_sizes['alpha'])))
            self.means[k] = self.means[k] + self.d_m[k]*self.step_sizes['m']
            self.precisions[k] = self.precisions[k] + self.d_invC[k]*self.step_sizes['invC']
            self.covariances[k] = inv(self.precisions[k])

    # TODO: insert bishop ref for r_nk - maybe clean it up a little bit (later)? Looks opaque
    def E_step(self, X):
        N = X.shape[0]
        for k in range(self.K):
            if self.alpha[k] <= self.ALPHA_LB:
                self.responsibilities[:, k] = np.zeros(N) # fix?
            else:
                for n in range(N):
                    self.responsibilities[n, k] = r_nk(k, self.alpha, self.means, self.covariances, self.inv_sigma, X[n])
                    
    def E_step_minibatch(self, X, alpha_lb=0.1):
        # xn = X[np.random.randint(X.shape[0])]
        pass

    def perturb_variational_params(self, non_diag=False):
        # fill in
        pass
                    
    def update_precision_from_covariance(self):
        for k in range(self.K):
            self.precisions[k] = inv(self.covariances[k])
            
            

# "priors"
alpha_0 = 1e-3
m_0 = np.zeros(2)
C_0 = np.eye(2)
K = 5

# Iterations and runtype
N_its = 40
# runtype = 'GD (perturbed M-step init)'
# runtype = 'GD (random init)'
runtype = 'CAVI'

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
def calculate_ELBO(variational_distribution, joint_distribution):
    """Based on equation 10.70-10.77 from Bishop, with some modifications due to our fixed covariance."""
    alpha0 = joint_distribution.alpha
    m0 = joint_distribution.mean
    C0 = joint_distribution.covariance
    alpha = variational_distribution.alpha
    m = variational_distribution.means
    C = variational_distribution.covariances
    r = variational_distribution.responsibilities
    NK = variational_distribution.NK
    xbar = variational_distribution.xbar
    SK = variational_distribution.SK
    invSig = variational_distribution.inv_sigma

    p1 = E_ln_p_X_given_Z_mu(m, invSig, C0, NK, xbar, SK, D=2)  # ~ Eqn 10.71
    p2 = E_ln_p_Z_given_pi(r, alpha)                            # = Eqn 10.72
    p3 = E_ln_p_pi(alpha0, alpha)                               # = Eqn 10.73
    p4 = E_ln_p_mu(m, C, m0, inv(C0), D=2)                      # ~ Eqn 10.74
    q1 = E_ln_q_Z(r)                                            # = Eqn 10.75
    q2 = E_ln_q_pi(alpha)                                       # = Eqn 10.76
    q3 = E_ln_q_mu(C, D=2)                                      # ~ Eqn 10.77
    return p1+p2+p3+p4-q1-q2-q3

variational_distribution_memory = []
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
    variational_distribution_memory.append(deepcopy(variational))
    variational.M_step(X, joint)
    variational.E_step(X)
    title = '(Class) %s: Iteration %d'%(variational_distribution_memory[-1].update_type, i)
    filename = 'plots/img%04d.png'%i
    Epi = E_pi(variational.alpha, joint.alpha, X.shape[0])
    plot_GMM(X, variational_distribution_memory[-1].means, K_inv_sigma, Epi, centres, covs, K, title,  savefigpath=filename)
    ELBO[i] = calculate_ELBO(variational, joint, inv_sigma) #TODO: fix inv sigma
    
# Make and display gif 
gifname = make_gif(filedir, gifdir)
# delete pngs for next run
for file in os.listdir(filedir):
  os.remove(os.path.join(filedir,file))
  
plt.close('all')
plt.figure()
plt.plot(ELBO)
plt.title('ELBO')
plt.xlabel('Iterations')
plt.show()
