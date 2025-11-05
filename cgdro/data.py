import numpy as np
#import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import multivariate_normal

class DataContainerSimu_linear_reg_lowd:
    def __init__(self, n_list, N, p=5):
        """
        n_list : list of sample sizes for each source domain
        N      : number of target samples
        p      : number of features (default=5)
        """
        self.n_list = n_list
        self.N = N
        self.p = p
        self.L = len(n_list)

        self.X_sources_list = []
        self.Y_sources_list = []
        self.X_target = None

        # parameters for simulation
        self.mean_source = np.zeros(p)
        self.cov_source = np.eye(p)

        self.mean_target = np.zeros(p)
        self.cov_target = np.eye(p)

        self.beta_list = []

    def generate_funcs_list(self, seed=None):
        np.random.seed(seed)
        self.beta_list = []

        # hard-coded example like your script
        b1 = np.zeros(self.p)
        b1[0:5] = np.arange(1, 6) / 20
        b2 = -np.arange(6, 1, -1) / 15
        b3 = np.array([0.175, -0.175, 0.175, -0.175, 0.175])

        self.beta_list = [b1, b2, b3]

    def generate_data(self, seed=None):
        np.random.seed(seed)
        self.X_sources_list = []
        self.Y_sources_list = []

        # source groups
        for l, n in enumerate(self.n_list):
            X = multivariate_normal.rvs(mean=self.mean_source,
                                        cov=self.cov_source,
                                        size=n)
            if l==1:
                Y = X @ self.beta_list[l] + np.random.normal(size=n) * 2
            else:
                Y = X @ self.beta_list[l] + np.random.normal(scale=0.5, size=n)
            self.X_sources_list.append(X)
            self.Y_sources_list.append(Y)

        # target group
        self.X_target = multivariate_normal.rvs(mean=self.mean_target,
                                                cov=self.cov_target,
                                                size=self.N)



class DataContainerSimu_linear_reg_highd:
    def __init__(self, n_list, N, p=100):
        """
        n_list : list of sample sizes for each source domain
        N      : number of target samples
        p      : number of features (default=100)
        """
        self.n_list = n_list
        self.N = N
        self.p = p
        self.L = len(n_list)

        self.X_sources_list = []
        self.Y_sources_list = []
        self.X_target = None

        # mean for sources
        self.mean_source = np.zeros(p)
        # AR(1)-like covariance for sources
        self.cov_source = self._A1gen(rho=0.6, p=p)

        # mean and covariance for target
        self.mean_target = np.zeros(p) + 0.1
        self.cov_target = self.cov_source.copy()
        for i in range(p):
            self.cov_target[i, i] = 1.5
        for i in range(5):
            for j in range(5):
                if i != j:
                    self.cov_target[i, j] = 0.9
        for i in range(98, 100):
            for j in range(98, 100):
                if i != j:
                    self.cov_target[i, j] = 0.9

        # regression coefficients
        self.beta_list = []

    def _A1gen(self, rho, p):
        A1 = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                A1[i, j] = rho ** abs(i - j)
        return A1

    def generate_funcs_list(self, seed=None):
        np.random.seed(seed)
        b1 = np.zeros(self.p)
        b1[0:5] = np.arange(1, 6) / 20
        b1[97:100] = [0.5, -0.5, -0.5]

        b2 = np.zeros(self.p)
        b2[5:10] = np.arange(1, 6) / 20
        b2[97:100] = 0.5 * np.array([0.5, -0.5, -0.5])

        self.beta_list = [b1, b2]

    def generate_data(self, seed=None):
        np.random.seed(seed)
        self.X_sources_list = []
        self.Y_sources_list = []

        # source groups
        for l, n in enumerate(self.n_list):
            X = multivariate_normal.rvs(mean=self.mean_source,
                                        cov=self.cov_source,
                                        size=n)
            Y = X @ self.beta_list[l] + np.random.normal(size=n)
            self.X_sources_list.append(X)
            self.Y_sources_list.append(Y)

        # target group
        self.X_target = multivariate_normal.rvs(mean=self.mean_target,
                                                cov=self.cov_target,
                                                size=self.N)





class DataContainerSimu_Nonlinear_reg:
    def __init__(self, n, N):
        self.n = n            # number of samples in each source domain
        self.N = N            # number of samples in the target domain
        self.d = 5            # number of features
        self.L = None         # number of source domains
        self.X_sources_list = []  # list of source covariate matrices
        self.Y_sources_list = []  # list of source outcome vectors
        self.X_target = None  # target covariate matrix
        self.Y_target_potential_list = []  # list of potential target outcome vectors
        self.f_funcs = []   # list of source conditional outcome functions
        self.mu0 = None       # target covariate distribution mean, used when generating data
        self.Sigma0 = None    # target covariate distribution covariance, used when generating data
        
    def generate_funcs_list(self, L, seed=None):
        np.random.seed(seed)
        self.L = L
        beta_list = []
        A_list = []
        self.mu0 = np.array([1, -1, 0.5, 0., 0.])
        X_sample = np.random.randn(20000, self.d) + self.mu0
        for l in range(L):
            # random beta in [-1,1]^d
            beta = np.random.uniform(-1, 1, self.d)
            beta_list.append(beta)
            
            # random symmetric matrix A
            B = np.random.uniform(-0.5, 0.5, size=(self.d, self.d))
            A = (B + B.T) / 2
            A_list.append(A)
            
            # compute c = trace(A) + mu^T A mu
            c = np.trace(A) + self.mu0.dot(A.dot(self.mu0))
            
            def f_func(x, beta=beta, A=A, c=c):
                return np.sin(x.dot(beta)) + np.sum(np.dot(x, A) * x, axis=1) - c
            self.f_funcs.append(lambda x, f_func=f_func: f_func(x) - np.mean(f_func(X_sample)))
    
    def generate_data(self, seed=None):
        np.random.seed(seed)
        
        # ------- Generate Source Data -------
        mu = np.zeros(self.d)
        Sigma = np.eye(self.d)
        for l in range(self.L):
            X = np.random.multivariate_normal(mu, Sigma, self.n)
            if l == 1:
                Y = self.f_funcs[l](X) + np.random.randn(self.n) * 3
            else:
                Y = self.f_funcs[l](X) + np.random.randn(self.n) * 0.5
            self.X_sources_list.append(X)
            self.Y_sources_list.append(Y)
        
        # ------- Generate Target Data -------
        self.Sigma0 = np.eye(self.d)
        self.X_target = np.random.multivariate_normal(self.mu0, self.Sigma0, self.N)
        for l in range(self.L):
            Y_target =  self.f_funcs[l](self.X_target) + np.random.randn(self.N)
            self.Y_target_potential_list.append(Y_target)
        

class DataContainerSimu_Nonlinear_reg_prior:
    def __init__(self, n, N, N_label):
        self.n = n            # number of samples in each source domain
        self.N = N            # number of samples in the target domain
        self.N_label = N_label # number of labeled samples in the target domain
        self.d = 5            # number of features
        self.L = None         # number of source domains
        self.X_sources_list = []  # list of source covariate matrices
        self.Y_sources_list = []  # list of source outcome vectors
        self.X_target = None  # target covariate matrix
        self.Y_target = None  # target outcome vector
        self.X_target_label = None 
        self.X_target_label = None
        self.f_funcs = []   # list of source conditional outcome functions
        self.mu0 = None       # target covariate distribution mean, used when generating data
        self.Sigma0 = None    # target covariate distribution covariance, used when generating data
        
    def generate_funcs_list(self, L=4, seed=None):
        np.random.seed(seed)
        self.L = L
        beta_list = []
        A_list = []
        self.mu0 = np.array([1, -1, 0.5, 0., 0.])
        X_sample = np.random.randn(20000, self.d) + self.mu0
        for l in range(L):
            # random beta in [-1,1]^d
            beta = np.random.uniform(-1, 1, self.d)
            beta_list.append(beta)
            
            # random symmetric matrix A
            B = np.random.uniform(-0.5, 0.5, size=(self.d, self.d))
            A = (B + B.T) / 2
            A_list.append(A)
            
            # compute c = trace(A) + mu^T A mu
            c = np.trace(A) + self.mu0.dot(A.dot(self.mu0))
            
            def f_func(x, beta=beta, A=A, c=c):
                return np.sin(x.dot(beta)) + np.sum(np.dot(x, A) * x, axis=1) - c
            self.f_funcs.append(lambda x, f_func=f_func: f_func(x) - np.mean(f_func(X_sample)))
    
    def generate_data(self, seed=None):
        np.random.seed(seed)
        self._reset_data_lists()
        
        # ------- Generate Source Data -------
        mu = np.zeros(self.d)
        Sigma = np.eye(self.d)
        for l in range(self.L):
            X = np.random.multivariate_normal(mu, Sigma, self.n)
            Y = self.f_funcs[l](X) + np.random.randn(self.n)
            self.X_sources_list.append(X)
            self.Y_sources_list.append(Y)
        
        # ------- Generate Target Data -------
        self.Sigma0 = np.eye(self.d)
        self.X_target = np.random.multivariate_normal(self.mu0, self.Sigma0, self.N)
        weights = np.concatenate([[0.6], 0.4 / np.ones(self.L - 1)])
        self.Y_target = np.random.randn(self.N)
        for l in range(self.L):
            self.Y_target += weights[l] * self.f_funcs[l](self.X_target)
        self.X_target_label = self.X_target[:self.N_label]
        self.Y_target_label = self.Y_target[:self.N_label]
    
    def _reset_data_lists(self):
        self.X_sources_list = []
        self.Y_sources_list = []

        

def softmax(x):
    """
    Row-wise softmax
    Input:  n × (C) or n × (C-1) matrix of logits
    Output: n × C matrix of probabilities
    """
    x_max = np.max(x, axis=1, keepdims=True)  # numerical stability
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class DataContainerSimu_linear_Cl:
    def __init__(self, n, N, p, L, K):
        """
        n : number of samples per source domain
        N : number of samples in target domain
        p : number of features
        L : number of source domains
        K : number of classes
        """
        self.n = n
        self.N = N
        self.p = p
        self.L = L
        self.K = K

        self.X_sources_list = []
        self.Y_sources_list = []
        self.X_target = None
        self.beta_list = []
        self.probs_list = []

    def generate_funcs_list(self, seed=None):
        np.random.seed(seed)
        self.beta_list = []
        for _ in range(self.L):
            # baseline column of zeros + random effects for K classes
            beta = np.column_stack((
                np.zeros(self.p), 
                np.random.normal(0, 0.25, (self.p, self.K))
            ))
            self.beta_list.append(beta)

    def generate_data(self, seed=None):
        np.random.seed(seed)
        self.X_sources_list = []
        self.Y_sources_list = []
        self.probs_list = []

        # ----- source domains -----
        for l in range(self.L):
            X = np.random.normal(0, 1, (self.n, self.p))
            logits = X.dot(self.beta_list[l])
            logits = logits - np.mean(logits)  # center
            probs = softmax(logits)
            y = np.array([
                np.random.multinomial(1, probs[i, :]).tolist().index(1)
                for i in range(self.n)
            ])
            self.X_sources_list.append(X)
            self.Y_sources_list.append(y)
            self.probs_list.append(probs)

        # ----- target domain (covariate shift only, no labels) -----
        self.X_target = np.random.normal(0.1, 1, (self.N, self.p))
