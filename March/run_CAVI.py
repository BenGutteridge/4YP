import numpy as np
from numpy.linalg import inv
from statistics_of_observed_data import N_k, x_k_bar, S_k
from CAVI_updates import alpha_k, m_invC_k
from generate_dataset import generate_2D_dataset
from calculate_responsibilities import r_nk
from plot_utils import plot_GMM, E_pi
from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.close('all')


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
        


alpha_0 = 1e-3
m_0 = np.zeros(2)
C_0 = np.eye(2)
K = 2
N = 500
N_its = 10

X, centres, covs = generate_2D_dataset(N, K=2, weights=np.array([0.1,0.9]))

# Fixed covariance of every component of the Gaussian mixture
sigma = inv_sigma = np.eye(2)
K_inv_sigma = [inv_sigma for _ in range(K)]

initial_responsibilities = np.random.dirichlet(np.ones(K), N)  # Size (N, K)

variational = VariationalDistribution(initial_responsibilities)
joint = JointDistribution(alpha_0, m_0, covariance=C_0)


def M_step(X, variational_distribution, joint_distribution):
    r = variational_distribution.responsibilities
    NK, xbar, SK = [],[],[]

    for k in range(K):
        Nk = N_k(r[:,k])
        xkbar = x_k_bar(Nk, r[:,k], X)
        Sk = S_k(Nk, r[:,k], X, xkbar)
    
        variational_distribution.alpha[k] = alpha_k(Nk, joint.alpha)
        variational_distribution.means[k], variational_distribution.precisions[k] = m_invC_k(Nk, xkbar, joint.mean, joint.precision, inv_sigma)
        variational_distribution.covariances[k] = inv(variational_distribution.precisions[k])
    
        NK.append(Nk)
        xbar.append(xkbar.reshape(2,))
        SK.append(Sk)
    return variational_distribution, NK, xbar, SK

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

variational_distribution_memory = []
for i in tqdm(range(N_its)):
    variational_distribution_memory.append(deepcopy(variational))
    variational, NK, xbar, SK = M_step(X, variational, joint)
    variational = E_step(X, variational, inv_sigma)
    
    title = '(Class) CAVI: %d'%i
    Epi = E_pi(variational.alpha, joint.alpha, X.shape[0])
    plot_GMM(X, variational_distribution_memory[-1].means, K_inv_sigma, Epi, centres, covs, K, title)

