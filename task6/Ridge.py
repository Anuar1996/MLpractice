from scipy.linalg import solve
import numpy as np

class RidgeRegression():
    
    def __init__(self, lambda_=0.1):
        self.lambda_ = lambda_
    
    def fit(self, X, y):
        n_features = X.shape[1]
        
        I = np.diag(np.ones(n_features))
        A = (X.T).dot(X) + self.lambda_ * I
        B = (X.T).dot(y)
        self.w_ = solve(A, B)
    
    def predict(self, X):
        return X.dot(self.w_)