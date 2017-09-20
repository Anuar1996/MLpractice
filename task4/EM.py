import numpy as np
import time
from math import log, pi, exp
from scipy.linalg import solve

class EM_algorithm(object):
    def __init__(self,
                 sigma_s=None,
                 mu_s=None,
                 pi_s=None,
                 cov_type='full',
                 components=10,
                 tol=1e-3,
                 max_iter=100):
        ''' 
            Expectation-Minimization algorithm is an 
            iterative method for finding maximum likelihood or maximum 
            a posteriori (MAP) estimates of parameters in statistical models. 

            X: numpy array with shape (N, D); 
            N - samples, D - features 

            sigma_s: numpy array with shape (K, D, D); 
            K - count of components, D - features 
            
            mu_s: numpy array with shape (K, D); 
            K - count of components, D - features 

            pi_s: numpy array with shape K; 
            K - count of components 
            
            cov_type: string;
            Type of covariance matrix
            
            components: int;
            Count of components
            
            tol: float (optional) 

            max_iter: int (optional) 
        '''
        
        if (cov_type not in ['diag', 'full']):
            raise ValueError('Covariance matrix must be diag or full')
        self.cov_type = cov_type
        
        if (max_iter <= 0):
            raise ValueError('max_iter must be positive')
        self.max_iter = max_iter
        if (components <= 0):
            raise ValueError('count of components must be positive')
        self.components = components
        if (sigma_s is not None):
            if (self.components != sigma_s.shape[0]):
                raise ValueError('count of sigma_s must be equal to components')
        self.sigma_s = sigma_s
        if (mu_s is not None):
            if (self.components != mu_s.shape[0]):
                raise ValueError('count of mu_s must be equal to components')
        self.mu_s = mu_s
        if (pi_s is not None):
            if (self.components != pi_s.shape[0]):
                raise ValueError('count of pi_s must be equal to components')
        self.pi_s = pi_s
        self.tol = tol
        
        
    def E_step(self, X):    
        if (len(X.shape)==1):
            X = X[:, np.newaxis]
        N, D = X.shape
        K = self.components
        log_normal = np.zeros((N, K))
        if (self.cov_type=='full'):
            for k in range(K):
                det_sigma = np.sum(np.log(np.diag(np.linalg.cholesky(self.sigma_s_new[k]))))
                log_normal[:, k] = -0.5 * D * log(2 * pi) - det_sigma - 0.5*\
                            np.sum((X - self.mu_s_new[k]) * solve(self.sigma_s_new[k], 
                                                   (X - self.mu_s_new[k]).T).T, axis=1)
        if (self.cov_type=='diag'):
            for k in range(K):
                sigma = np.diag(self.sigma_s_new[k])
                log_normal[:, k] = -0.5 * D * log(2 * pi) + 0.5 * np.sum(np.log(sigma)) - 0.5 * np.sum(((X - self.mu_s_new[k]) ** 2 / sigma), axis=1)
        self.log_max = np.max(log_normal, axis=1)[:, np.newaxis]
        self.gamma = self.pi_s_new * np.exp(log_normal - self.log_max)
        
    def fit(self, X):
        if (len(X.shape)==1):
            X = X[:, np.newaxis]
        N, D = X.shape
        K = self.components
        
        if (self.pi_s is None):
            self.pi_s = np.zeros(K)
            for k in range(K):
                self.pi_s[k] = 1 / K
        
        if (self.sigma_s is None):
            self.sigma_s = np.zeros((K, D, D))
            for k in range(K):
                self.sigma_s[k] = np.diag(np.ones(D))
        
        if (self.mu_s is None):
            self.mu_s = X[np.random.randint(0, N, K)]
        
        self.pi_s_new = self.pi_s
        self.mu_s_new = self.mu_s
        self.sigma_s_new = self.sigma_s
        
        self.gamma = np.zeros((N, K))
        iteration = 0
    
        log_old = 0
        log_new = 1
        likelihood = []
        t = []
        self.E_step(X)
        
        while(np.abs(log_old - log_new) > self.tol):
            start = time.clock()
        
            #E-step; find latent variables gamma:
            self.gamma /= self.gamma.sum(axis=1)[:, np.newaxis]
            
            #M-step; find new mu_s, sigma_s, pi_s
            N_k = self.gamma.sum(axis=0)
            self.mu_s_new = (self.gamma.T).dot(X) / N_k[:, np.newaxis]
            
            self.pi_s_new = N_k / N
            
            if(self.cov_type=='full'):
                for k in range(K): 
                    self.sigma_s_new[k] = np.dot(self.gamma[:, k] * (X - self.mu_s_new[k]).T, 
                                             X - self.mu_s_new[k])
            if (self.cov_type=='diag'):
                for k in range(K):
                    self.sigma_s_new[k] = np.diag(np.sum((self.gamma[:, k, np.newaxis] * 
                             (X - self.mu_s_new[k]) ** 2), axis=0))
            
            self.sigma_s_new += np.eye(D)[np.newaxis, :, :] * 1e-4
            self.sigma_s_new /= N_k[:, np.newaxis, np.newaxis]
        
            status = (iteration >= self.max_iter)
            if (status):
                break
        
            #log likelihood
            log_old = log_new
            
            #this is E-step
            self.E_step(X)
            log_new = (np.log(np.sum(self.gamma, axis=1)) + self.log_max.ravel()).sum()
            likelihood += [log_new / N]
            
            t += [time.clock() - start]
            iteration += 1
            
        self.res = {
                    'sigma': self.sigma_s_new,
                    'gamma': self.gamma,
                    'mu': self.mu_s_new,
                    'pi': self.pi_s_new,
                    't': t,
                    'likelihood': likelihood
                }
    
    def likelihood(self, X):
        if (len(X.shape)==1):
            X = X[:, np.newaxis]
        N, D = X.shape
        self.E_step(X)
        return np.log(np.sum(self.gamma, axis=1)) + self.log_max.ravel()
