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
            
    def M_step(self, X, joint_distribution, inv_sigma):
        self.calculate_Nk_xkbar_Sk(X)
        for k in range(self.K):
            self.alpha[k] = alpha_k(self.NK[k], joint.alpha)
            self.means[k], self.precisions[k] = m_invC_k(self.NK[k], self.xbar[k], joint.mean, joint.precision, inv_sigma)
            self.covariances[k] = inv(self.precisions[k])
        self.update_type = 'CAVI'
            
    def M_step_GD(self, X, joint_distribution, invSig,
                  step_sizes={'alpha': 1.0, 'm': 1e-2, 'invC': 1e-4}):
        # performing GD update for M step rather than CAVI update equations
        self.step_sizes = step_sizes
        
        self.calculate_Nk_xkbar_Sk(X)
        
        # Gradient update equations
        self.d_alpha = L_grad_alpha(joint_distribution.alpha, self.alpha, self.NK)
        self.d_m = L_grad_m(self.means, joint_distribution.mean, joint_distribution.precision, invSig, self.NK, self.xbar)
        self.d_invC = L_grad_invC(self.means, self.precisions, invSig, joint_distribution.precision, self.NK)
    
        for k in range(self.K): 
            # constraints: alpha>0. Setting alpha>.1 as psi'(alpha->0) -> inf
            self.alpha[k] = np.max((0.1, self.alpha[k] + self.d_alpha[k]*step_sizes['alpha']))
            self.means[k] = self.means[k] + self.d_m[k]*step_sizes['m']
            self.precisions[k] = self.precisions[k] + self.d_invC[k]*step_sizes['invC']
            self.covariances[k] = inv(self.precisions[k])
        self.update_type = 'GD'
            
    def E_step(self, X, inv_sigma, alpha_lb=0.1):
        N = X.shape[0]
        for k in range(self.K):
            if self.alpha[k] <= alpha_lb: 
                self.responsibilities[:,k] = np.zeros(N) # fix?
            else:
                for n in range(N):
                    self.responsibilities[n,k] = r_nk(k, self.alpha, self.means, self.covariances, inv_sigma, X[n])
                    
    def E_step_minibatch(self, X, inv_sigma, alpha_lb=0.1):
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

# Iterations
N_its = 40

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

variational = VariationalDistribution(initial_responsibilities)
joint = JointDistribution(alpha_0, m_0, covariance=C_0)


def M_step(X, variational_distribution, joint_distribution):
    q = variational_distribution
    q.calculate_Nk_xkbar_Sk(X)

    for k in range(K):
        q.alpha[k] = alpha_k(q.NK[k], joint.alpha)
        q.means[k], q.precisions[k] = m_invC_k(q.NK[k], q.xbar[k], joint.mean, joint.precision, inv_sigma)
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

def M_step_GD(X, variational_distribution, joint_distribution, invSig,
              step_sizes={'alpha': 1.0, 'm': 1e-2, 'invC': 1e-4}):
    # performing GD update for M step rather than CAVI update equations
    q = variational_distribution
    p = joint_distribution
    q.calculate_Nk_xkbar_Sk(X)
    
    # Gradient update equations
    d_alpha = L_grad_alpha(p.alpha, q.alpha, q.NK)
    d_m = L_grad_m(q.means, p.mean, p.precision, invSig, q.NK, q.xbar)
    d_invC = L_grad_invC(q.means, q.precisions, invSig, p.precision, q.NK)

    for k in range(q.K): 
        # constraints: alpha>0. Setting alpha>.1 as psi'(alpha->0) -> inf
        q.alpha[k] = np.max((0.1, q.alpha[k] + d_alpha[k]*step_sizes['alpha']))
        q.means[k] = q.means[k] + d_m[k]*step_sizes['m']
        q.precisions[k] = q.precisions[k] + d_invC[k]*step_sizes['invC']
        q.covariances[k] = inv(q.precisions[k])
        
    return variational_distribution, 'GD'


def calculate_ELBO(variational_distribution, joint_distribution, invSig):
    # certainly not the neatest way to do everything, but better than rewriting all my functions
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


variational.M_step(X, joint, inv_sigma)
variational.alpha, variational.means, variational.covariances = perturb_variational_params(alpha=variational.alpha, 
                                                                                            m=variational.means, 
                                                                                            C=variational.covariances, 
                                                                                            non_diag=True)
variational.update_precision_from_covariance() # need to make sure both change whenever one does

for i in tqdm(range(N_its)):
    variational_distribution_memory.append(deepcopy(variational))
    # variational = M_step(X, variational, joint)
    # variational.M_step(X, joint, inv_sigma)
    # variational, update_type = M_step_GD(X, variational, joint, inv_sigma)
    variational.M_step_GD(X, joint, inv_sigma)
    # variational = E_step(X, variational, inv_sigma)
    variational.E_step(X, inv_sigma)
    
    title = '(Class) %s: Iteration %d'%(variational_distribution_memory[-1].update_type, i)
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

