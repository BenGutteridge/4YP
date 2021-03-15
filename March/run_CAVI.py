import numpy as np
from numpy.linalg import inv
from statistics_of_observed_data import N_k, x_k_bar, S_k
from CAVI_updates import alpha_k, m_invC_k
from generate_dataset import generate_2D_dataset
from calculate_responsibilities import r_nk
from plot_utils import plot_GMM, E_pi, make_gif
from calculate_ELBO import E_ln_p_X_given_Z_mu, E_ln_p_Z_given_pi, E_ln_p_pi, E_ln_p_mu
from calculate_ELBO import E_ln_q_Z, E_ln_q_pi, E_ln_q_mu
from copy import deepcopy
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
plt.close('all')
plt.ioff()


class JointDistribution:
    def __init__(self, alpha, mean, covariance=None, precision=None):
        """

        :param alpha: Parameter of Dirichlet distribution for mixture weights
        :param means:
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
    def __init__(self, initial_responsibilities):
        self.responsibilities = initial_responsibilities
        self.K = initial_responsibilities.shape[1]
        self.D = 2
        self.alpha = np.zeros(self.K)
        self.means = np.zeros((self.K,self.D))
        self.precisions = self.covariances = np.zeros((self.K,self.D,self.D))
        
    def calculate_Nk_xkbar_Sk(self, X):
        r = self.responsibilities
        self.NK = np.zeros(self.K)
        self.xbar = np.zeros((self.K,self.D))
        self.SK = np.zeros((self.K,self.D,self.D))
        for k in range(self.K):
            self.NK[k] = N_k(r[:,k])
            self.xbar[k] = x_k_bar(self.NK[k], r[:,k], X)
            self.SK[k] = S_k(self.NK[k], r[:,k], X, self.xbar[k])
            

# "priors"
alpha_0 = 1e-3
m_0 = np.zeros(2)
C_0 = np.eye(2)
K = 3

# Iterations
N_its = 5

# Generate dataset
N = 100
num_clusters = 2
X, centres, covs = generate_2D_dataset(N, K=num_clusters, 
                                       weights=np.array([0.1,0.9]),
                                       )

# Fixed covariance of every component of the Gaussian mixture
sigma = inv_sigma = np.eye(2)
K_inv_sigma = [inv_sigma for _ in range(K)]

initial_responsibilities = np.random.dirichlet(np.ones(K), N)  # Size (N, K)

variational = VariationalDistribution(initial_responsibilities)
joint = JointDistribution(alpha_0, m_0, covariance=C_0)


def M_step(X, variational_distribution, joint_distribution):
    q = variational_distribution
    q.calculate_Nk_xkbar_Sk(X)

    for k in range(K):
        q.alpha[k] = alpha_k(q.NK[k], joint.alpha)
        q.means[k], variational_distribution.precisions[k] = m_invC_k(q.NK[k], q.xbar[k], joint.mean, joint.precision, inv_sigma)
        q.covariances[k] = inv(q.precisions[k])
    
    return variational_distribution

def E_step(X, variational_distribution, inv_sigma, alpha_lb=0.1):
    N = X.shape[0]
    q = variational_distribution
    for k in range(variational_distribution.K):
        if q.alpha[k] <= alpha_lb: 
            q.responsibilities[:,k] = np.zeros(N) # fix?
        else:
            for n in range(N):
                q.responsibilities[n,k] = r_nk(k, q.alpha, q.means, q.covariances, inv_sigma, X[n])
    return q


def calculate_ELBO(variational_distribution, joint_distribution, invSig):
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
    
    p1 = E_ln_p_X_given_Z_mu(m, invSig, C0, NK, xbar, SK, D=2)
    p2 = E_ln_p_Z_given_pi(r, alpha)
    p3 = E_ln_p_pi(alpha0, alpha)
    p4 = E_ln_p_mu(m, C, m0, inv(C0), D=2)
    q1 = E_ln_q_Z(r)
    q2 = E_ln_q_pi(alpha)
    q3 = E_ln_q_mu(C, D=2)    
    return p1+p2+p3+p4-q1-q2-q3

variational_distribution_memory = []
ELBO = np.zeros(N_its)
# Saving plots and making gif
filedir = 'plots'
gifdir = 'gifs'


for i in tqdm(range(N_its)):
    variational_distribution_memory.append(deepcopy(variational))
    variational = M_step(X, variational, joint)
    variational = E_step(X, variational, inv_sigma)
    
    title = '(Class) CAVI: %d'%i
    filename = 'plots/img%04d.png'%i
    Epi = E_pi(variational.alpha, joint.alpha, X.shape[0])
    plot_GMM(X, variational_distribution_memory[-1].means, K_inv_sigma, Epi, centres, covs, K, title,  savefigpath=filename)
    ELBO[i] = calculate_ELBO(variational, joint, inv_sigma)
    
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

