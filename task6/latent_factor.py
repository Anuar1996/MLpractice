import numpy as np
from scipy.linalg import solve

def dict_user_movie(X):
    user_movie = dict()
    user = np.unique(X[:, 0]).astype('int')
    movie = np.unique(X[:, 1]).astype('int')
    for userID in user:
        indx_u = np.where(X[:, 0] == userID)[0].astype('int')
        indx_m = X[indx_u, 1].astype('int')
        r_ = X[indx_u, 2]
        user_movie[userID] = (indx_m, indx_u, r_)
    movie_user = dict()
    for movieID in movie:
        indx_m = np.where(X[:, 1] == movieID)[0].astype('int')
        indx_u = X[indx_m, 0].astype('int')
        movie_user[movieID] = (indx_u, indx_m)
    
    return (user_movie, movie_user)

class LatentFactor():
    
    def __init__(self, K=10, max_iter=20, lambda_p=0.2, lambda_q=0.001):
        self.K = K
        self.max_iter = max_iter
        self.lambda_p = lambda_p
        self.lambda_q = lambda_q
    
    def fit(self, user_movie, movie_user, y):
        
        num_users = len(user_movie.keys())
        num_movies = len(movie_user.keys())
        
        self.P = 0.01 * np.random.random(size=(num_users, self.K)) #P
        self.Q = 0.01 * np.random.random(size=(max(movie_user.keys()) + 1, self.K)) #Q
        tmpID = np.zeros((max(movie_user.keys()) + 1)) #index - movie ID
        tmpID[movie_user.keys()] = np.arange(num_movies).astype('int')
        
        for i in range(self.max_iter):
            for userID in user_movie.keys():
                indx_u = user_movie[userID][1]
                indx_m = user_movie[userID][0]
                r_u = y[indx_u]
                n_u = indx_m.shape[0]

                Q_u = self.Q[indx_m]
                A_u = (Q_u.T).dot(Q_u)
                d_u = (Q_u.T).dot(r_u)
                I = np.diag(np.ones(A_u.shape[0]))
                self.P[userID] = solve(self.lambda_p * n_u * I + A_u, d_u)
                
            for movieID in movie_user.keys():
                indx_m = movie_user[movieID][1]
                indx_u = movie_user[movieID][0]
                r_i = y[indx_m]
                n_i = indx_u.shape[0]

                P_i = self.P[indx_u]
                A_i = (P_i.T).dot(P_i)
                d_i = (P_i.T).dot(r_i)
                I = np.diag(np.ones(A_i.shape[0]))
                self.Q[movieID] = solve(self.lambda_q * n_i * I + A_i, d_i)
                
    def predict(self, X, user_movie, movie_user):
        y = np.zeros(X.shape[0])
        for userID in user_movie.keys():
            indx_u = user_movie[userID][1]
            indx_m = user_movie[userID][0] 
            y[indx_u] = self.P[userID].dot(self.Q[indx_m].T)
                
        return y