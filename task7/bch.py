import numpy as np
import gf
import itertools

class BCH():
    
    def __init__(self, n, t):
        self.n = n
        self.t = t
        
        q = int(np.log2(n + 1))
        primpoly = open("primpoly.txt").readlines()
        primpoly = primpoly[0].replace(" ", "").split(',')
        primpoly = [int(poly) for poly in primpoly]
        primpoly_deg = np.log2(np.array(primpoly)).astype('int')
        prim_poly = primpoly[np.where(primpoly_deg >= q)[0][0]]
        self.pm = gf.gen_pow_matrix(prim_poly)
        
        self.R = [self.pm[i % self.pm.shape[0] - 1, 1] for i in range(1, 2 * self.t + 1)]
        self.R = np.array(self.R)
        
        
        self.g, _ = gf.minpoly(self.R, self.pm)
        self.m = self.g.shape[0] - 1
    
    def encode(self, U):
        mesg_cnt, k = U.shape
        
        x_m = np.zeros(self.m + 1).astype('int')
        x_m[0] = 1
        
        V = np.zeros((mesg_cnt, self.n)).astype('int')
        for i in range(mesg_cnt):
            u = U[i, :]
            
            x_m_prod_u = gf.polyprod(x_m, u, self.pm)
            div, mod = gf.polydivmod(x_m_prod_u, self.g, self.pm)
            
            v = gf.polyadd(x_m_prod_u, mod)
            ind = self.n - v.shape[0]
            V[i, ind:] = gf.polyadd(x_m_prod_u, mod)
        
        return V
    
    def decode(self, W, method='euclid'):
        
        sindrom = np.zeros((W.shape[0], 2*self.t)).astype('int')
        V_hat = np.zeros((W.shape)).astype('int')
        error = []
        
        
        for i in range(W.shape[0]):
            sindrom[i, :] = gf.polyval(W[i, :], self.R, self.pm)
            if np.all(sindrom[i, :] == 0):
                V_hat[i, :] = W[i, :]
            else:
                error.append(i)
        
        
        for i in error:
            if method == 'euclid':
                # sindrom polynom
                sindrom_poly = np.zeros(2*self.t + 1).astype('int')
                s = sindrom[i, :]
                s = s[::-1]
                sindrom_poly[: -1] = s
                sindrom_poly[-1] = 1
                
                # z^(2t + 1)
                z = np.zeros(2*self.t + 2).astype('int')
                z[0] = 1
                
                #euclid algorithm
                r, A, loc_poly_coef = gf.euclid(z, sindrom_poly, 
                                                self.pm, max_deg=self.t)
                
            elif method == 'pgz':
                s = sindrom[i, :]
                for errors_n in range(self.t, 0, -1):
                    
                    # matrix for linear solve
                    S = np.zeros((errors_n, errors_n)).astype('int')
                    for j in range(errors_n):
                        S[j, :] = s[j: j + errors_n]
                    b = s[errors_n: 2 * errors_n]
                    lambda_ = gf.linsolve(S, b,self.pm)
                    if np.all(np.isnan(lambda_)):
                        continue
                    else:
                        break
                        
                # decode error
                if (errors_n == 1) and np.all(np.isnan(lambda_)):
                    V_hat[i, :] = np.ones(self.n) * np.nan
                    continue
                    
                # coef of locator polynoms
                loc_poly_coef = np.zeros(lambda_.shape[0] + 1).astype('int')
                loc_poly_coef[-1] = 1
                loc_poly_coef[: -1] = lambda_
                    
            # find root
            locator_val = gf.polyval(loc_poly_coef, self.pm[:, 1], self.pm)
            roots = self.pm[np.where(locator_val == 0)[0], 1]
            pos_error = (-self.pm[roots - 1, 0]) % self.pm.shape[0]
            pos_error = self.n - pos_error - 1
            #error polynom
            error_poly = np.zeros(self.n).astype('int')
            error_poly[pos_error] = 1
                
            #decode
            v_hat = W[i, :] ^ error_poly
            s_v_hat = gf.polyval(v_hat, self.R, self.pm)
            
            if not np.all(s_v_hat == 0):
                V_hat[i, :] = np.ones(self.n) * np.nan
                continue
                
            if (roots.shape[0] != loc_poly_coef.shape[0] - 1):
                V_hat[i, :] = np.ones(self.n) * np.nan
                continue
            V_hat[i, :] = v_hat
                
        return V_hat
            
    def dist(self):
        
        U = np.array(list(itertools.product([0, 1], 
                    repeat=self.n - self.m)), dtype=int)[1: ]
        V = self.encode(U)
        res = np.min(np.sum(V, axis=1))
        
        return res