# Модуль, который хранит класс SVM

import numpy as np
import cvxopt as cv
from matplotlib import pyplot
from sklearn import svm
import time
from numpy import linalg
from scipy.spatial.distance import cdist


#Линейное ядро
def linear_kernel(x1, x2):
    return x1.dot(x2.T)

#Ядро rbf
def rbf_kernel(x1, x2, gamma=0):
    if ((len(x1.shape) == 2) and (len(x2.shape) == 2)):
        res = np.exp(-gamma * cdist(x1, x2) ** 2)
        return res
    elif ((len(x1.shape) == 1) and (len(x2.shape) == 1)):
        return np.exp(-gamma * linalg.norm(x1 - x2) ** 2)
    else:
        raise TypeError('x1 and x2 must be 1d or 2d arrays')

class SVM(object):
    def __init__(self, 
                 kernel="linear_kernel",
                 method='libsvm',
                 gamma=0.0,
                 verbose=False,
                 max_iter=1000,
                 C=1.0,
                 tol=1e-6,
                 alpha=None,
                 beta=None,
                 stochastic=None,
                 k=None):
        
        """
        Классификатор для решения задачи SVM
        
        Параметры:
        ----------------
        kernel: string (default='linear_kernel')
        Выбор ядра, используемого для решения задачи
        
        method: string (default='libsvm')
        Выбор метода решения задачи
        
        gamma: float, optional (default=0)
        Ширина rbf-ядра
        
        verbose: bool, optional (default=False)
        Отладочная информация в виде: номер итерации, значение целевой функции
        
        max_iter: int, optional (default=100)
        Максимальное количество итераций
        
        C: float, optional (default=1.0)
        Штраф за неправильно классифицируемые объекты
        
        tol: float, optional (default=1e-6)
        Точность
        
        alpha, beta: float (default=None)
        Параметры выбора шага в методе субградиентного спуска
        
        stochastic: bool (default=None)
        Стохастический метод субградиентного спуска
        
        k: int (default=None)
        Размер подвыборки, используемой в стохастическом варианте метода субградиентного спуска
        """
        
        self.kernel = kernel
        self.C = C
        self.method = method
        self.gamma = gamma
        self.verbose = verbose
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = None
        self.beta = None
        if (kernel == linear_kernel and gamma != 0):
            raise ValueError('gamma must be zero')
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        if ((alpha is not None) and (method != 'subgradient') 
                                and (method != 'subgradient_1')
                                and (method != 'subgradient_2')):
            raise ValueError('alpha must be None')
        if ((beta is not None)  and (method != 'subgradient') 
                                and (method != 'subgradient_1')
                                and (method != 'subgradient_2')):
            raise ValueError('beta must be None')
        if ((stochastic is not None) and (method != 'subgradient') 
                                     and (method != 'subgradient_1')
                                     and (method != 'subgradient_2')):
            raise ValueError('stochastic variant in subgradient method')
        if ((stochastic is not None) and (k is None)):
            raise ValueError('k must be nonzero because "k" is size of subsample')
        self.k = k
        self.stochastic = stochastic
    
    def svm_qp_primal_solver(self, X, y, C, tol=1e-6, 
                             max_iter=100, verbose=False):
       
        start = time.clock()
        
        cv.solvers.options['show_progress'] = verbose
        cv.solvers.options['maxiters'] = max_iter
        cv.solvers.options['abstol'] = tol
    
        N, D = X.shape
    
        #матрица P
        P_block_1 = cv.matrix(np.diag(np.ones(D)))
        P_block_2 = cv.matrix(np.zeros((D, N + 1)))
        P_block_3 = cv.matrix(np.zeros((N + 1, D)))
        P_block_4 = cv.matrix(np.zeros((N + 1, N + 1)))
        P_top = cv.matrix([[P_block_1], [P_block_2]])
        P_bottom = cv.matrix([[P_block_3], [P_block_4]])
        P = cv.matrix([P_top, P_bottom])
    
        #вектор q
        q_block_1 = cv.matrix(np.zeros(D))
        q_block_2 = cv.matrix(C * np.ones(N))
        q_block_3 = cv.matrix([0])
        q = cv.matrix([q_block_1, q_block_2, q_block_3])
    
        #матрица G
        G_block_1 = cv.matrix(-y[:,np.newaxis] * X)
        G_block_2 = cv.matrix(-np.diag(np.ones(N)))
        G_block_3 = cv.matrix(-y.astype('double'))
        G_block_4 = cv.matrix(np.zeros((N, D)))
        G_block_5 = cv.matrix(-np.diag(np.ones(N)))
        G_block_6 = cv.matrix(np.zeros(N))
        G_top = cv.matrix([[G_block_1], [G_block_2], [G_block_3]])
        G_bottom = cv.matrix([[G_block_4], [G_block_5], [G_block_6]])
        G = cv.matrix([G_top, G_bottom])
    
        #вектор h
        h_block_1 = cv.matrix(-np.ones(N))
        h_block_2 = cv.matrix(np.zeros(N))
        h = cv.matrix([h_block_1, h_block_2])
    
        solution = cv.solvers.qp(P, q, G, h)
        self.t = time.clock() - start
        
        self.a = None
        self.w = np.asarray(solution['x'][0: D]).reshape(D)
        self.b = solution['x'][N + D]
    
        self.status = solution['status'] == 'unknown'
        
        self.compute_primal_objective(X, y)
        
        self.res = {'w' : self.w,
                    'status': self.status,
                    'time': self.t,
                    'b' : self.b,
                    'objective': self.objective 
                   }
    
    def svm_qp_dual_solver(self, X, y, C, tol=1e-6, 
                           max_iter=100, verbose=False, gamma=0):
        start = time.clock()
        
        N, D = X.shape    
    
        cv.solvers.options['show_progress'] = verbose
        cv.solvers.options['maxiters'] = max_iter
        cv.solvers.options['abstol'] = tol
    
        #матрица P
        if (self.kernel == 'linear_kernel'):
            K = linear_kernel(X, X)
        if (self.kernel == 'rbf_kernel'):
            K = rbf_kernel(X, X, gamma=gamma)
        P = cv.matrix(np.outer(y, y) * K)
    
        #вектор q
        q = cv.matrix(-np.ones(N))
    
        #матрица G
        G_top = cv.matrix(np.diag(np.ones(N)))
        G_bottom = cv.matrix(-np.diag(np.ones(N)))
        G = cv.matrix([G_top, G_bottom])
    
        #вектор h
        h_block_1 = cv.matrix(C * np.ones(N))
        h_block_2 = cv.matrix(np.zeros(N))
        h = cv.matrix([h_block_1, h_block_2])
    
        #матрица A
        A = cv.matrix(y.astype('double'), (1, N))
    
        #вектор b
        b = cv.matrix(0.0)
        
        solution = cv.solvers.qp(P, q, G, h, A, b)
        self.t = time.clock() - start
    
        self.a = np.asarray(solution['x']).reshape(N)
        self.compute_support_vectors(X, y)
        
        ind = np.arange(len(self.a))[self.indices]
        self.b = 0.0
        for i in range(len(self.s_vectors_A)):
            self.b += self.s_vectors_y[i]
            self.b -= np.sum(self.s_vectors_A * self.s_vectors_y * K[ind[i], self.indices])
        self.b /= len(self.s_vectors_A)
        
        self.compute_w(X, y)
        self.status = solution['status'] == 'unknown'
        if (self.kernel == 'rbf_kernel'):
            self.compute_dual_objective()
        else:
            self.compute_primal_objective(X, y)
        
        self.res = {'w': self.w,
                   'status': self.status,
                   'time': self.t,
                   'support_vectors': self.s_vectors_X,
                   'a': self.a,
                   'b': self.b,
                   'objective': self.objective
        }
    def svm_subgradient_solver(self, X, y, C, tol=1e-6, max_iter=100, 
                               verbose=False, alpha=1e-2, beta=None, stochastic=None, k=None):
        start = time.clock()
        N, D = X.shape
        if (stochastic is not None):
            indices = np.random.choice(np.arange(N), k, replace=False)
            X = X[indices]
            y = y[indices]
        
        w_0 = np.ones(D + 1) 
        w_1 = np.ones(D + 1) + 1
        w_0 /= linalg.norm(w_0)
        w_1 /= linalg.norm(w_1)
        w_old = w_0[: D]
        b_old = w_0[D]
        w_new = w_1[: D]
        b_new = w_1[D]
        f_0 = 0.0
        f_1 = 1.0
        
        step = alpha
        self.objective_curve = []
        iteration = 0
        self.status = False
        
        while((np.abs(f_0 - f_1) >=self.tol) or (linalg.norm(w_0 - w_1) >= self.tol)):
            indices = y * (w_old.dot(X.T) + b_old) < 1
            sub_grad_w = w_old - C * np.sum((y[:,np.newaxis] * X)[indices])
            sub_grad_b = -C * np.sum(y[indices])
            w_new = w_old - step * sub_grad_w / linalg.norm(w_0)
            b_new = b_old - step * sub_grad_b / linalg.norm(w_0)
            w_0[:D] = w_old
            w_0[D] = b_old
            self.w = w_old
            self.b = b_old
            f_0 = self.compute_primal_objective(X, y)
            self.w = w_new
            self.b = b_new
            f_1 = self.compute_primal_objective(X, y)
            w_old = w_new
            b_old = b_new
            w_1[:D] = w_new
            w_1[D] = b_new
            self.objective_curve += [f_1]
            iteration += 1
            if (verbose):
                print(iteration, ": ", objective_curve[iteration - 1])
            if (beta == None):
                step = alpha
            else:
                step = alpha / (iteration ** beta)
            if (iteration == max_iter):
                self.status = True
                break
        self.t = time.clock() - start
        
        self.a = None
        self.w = w_new
        self.b = b_new
        self.res = {'w': self.w,
                    'b': self.b,
                    'status': self.status,
                    'time': self.t,
                    'objective_curve': self.objective_curve,
                    'objective': self.objective_curve[-1]
        }
        
    def svm_subgradient_solver_1(self, X, y, C, tol=1e-6, max_iter=100, 
                               verbose=False, alpha=1e-2, beta=None, stochastic=None, k=None):
        start = time.clock()
        N, D = X.shape
        if (stochastic is not None):
            indices = np.random.choice(np.arange(N), k, replace=False)
            X = X[indices]
            y = y[indices]
        
        w_0 = np.ones(D + 1) 
        w_1 = np.ones(D + 1) + 1
        w_0 /= linalg.norm(w_0)
        w_1 /= linalg.norm(w_1)
        w_old = w_0[: D]
        b_old = w_0[D]
        w_new = w_1[: D]
        b_new = w_1[D]
        f_0 = 0.0
        f_1 = 1.0
        
        step = alpha
        self.objective_curve = []
        iteration = 0
        self.status = False
        
        while((np.abs(f_0 - f_1) >= self.tol)):  #or (linalg.norm(w_0 - w_1) >= self.tol)
            indices = y * (w_old.dot(X.T) + b_old) < 1
            sub_grad_w = w_old - C * np.sum((y[:,np.newaxis] * X)[indices])
            sub_grad_b = -C * np.sum(y[indices])
            w_new = w_old - step * sub_grad_w / linalg.norm(w_0)
            b_new = b_old - step * sub_grad_b / linalg.norm(w_0)
            w_0[:D] = w_old
            w_0[D] = b_old
            self.w = w_old
            self.b = b_old
            f_0 = self.compute_primal_objective(X, y)
            self.w = w_new
            self.b = b_new
            f_1 = self.compute_primal_objective(X, y)
            w_old = w_new
            b_old = b_new
            w_1[:D] = w_new
            w_1[D] = b_new
            self.objective_curve += [f_1]
            iteration += 1
            if (verbose):
                print(iteration, ": ", objective_curve[iteration - 1])
            if (beta == None):
                step = alpha
            else:
                step = alpha / (iteration ** beta)
            if (iteration == max_iter):
                self.status = True
                break
        self.t = time.clock() - start
        
        self.a = None
        self.w = w_new
        self.b = b_new
        self.res = {'w': self.w,
                    'b': self.b,
                    'status': self.status,
                    'time': self.t,
                    'objective_curve': self.objective_curve,
                    'objective': self.objective_curve[-1]
        }

    def svm_subgradient_solver_2(self, X, y, C, tol=1e-6, max_iter=100, 
                               verbose=False, alpha=1e-2, beta=None, stochastic=None, k=None):
        start = time.clock()
        N, D = X.shape
        if (stochastic is not None):
            indices = np.random.choice(np.arange(N), k, replace=False)
            X = X[indices]
            y = y[indices]
        
        w_0 = np.ones(D + 1) 
        w_1 = np.ones(D + 1) + 1
        w_0 /= linalg.norm(w_0)
        w_1 /= linalg.norm(w_1)
        w_old = w_0[: D]
        b_old = w_0[D]
        w_new = w_1[: D]
        b_new = w_1[D]
        f_0 = 0.0
        f_1 = 1.0
        
        step = alpha
        self.objective_curve = []
        iteration = 0
        self.status = False
        
        while((linalg.norm(w_0 - w_1) >= self.tol)):
            indices = y * (w_old.dot(X.T) + b_old) < 1
            sub_grad_w = w_old - C * np.sum((y[:,np.newaxis] * X)[indices])
            sub_grad_b = -C * np.sum(y[indices])
            w_new = w_old - step * sub_grad_w / linalg.norm(w_0)
            b_new = b_old - step * sub_grad_b / linalg.norm(w_0)
            w_0[:D] = w_old
            w_0[D] = b_old
            self.w = w_old
            self.b = b_old
            f_0 = self.compute_primal_objective(X, y)
            self.w = w_new
            self.b = b_new
            f_1 = self.compute_primal_objective(X, y)
            w_old = w_new
            b_old = b_new
            w_1[:D] = w_new
            w_1[D] = b_new
            self.objective_curve += [f_1]
            iteration += 1
            if (verbose):
                print(iteration, ": ", objective_curve[iteration - 1])
            if (beta == None):
                step = alpha
            else:
                step = alpha / (iteration ** beta)
            if (iteration == max_iter):
                self.status = True
                break
        self.t = time.clock() - start
        
        self.a = None
        self.w = w_new
        self.b = b_new
        self.res = {'w': self.w,
                    'b': self.b,
                    'status': self.status,
                    'time': self.t,
                    'objective_curve': self.objective_curve,
                    'objective': self.objective_curve[-1]
        }
    
    def svm_liblinear_solver(self, X, y, C, tol=1e-6, 
                             max_iter=100, verbose=False):
        start = time.clock()
        N, D = X.shape
    
        s = svm.LinearSVC(C=C, tol=tol, max_iter=max_iter, verbose=verbose)
        s.fit(X, y)
        self.t = time.clock() - start
        
        self.w = s.coef_.reshape(D)
        self.b = s.intercept_
        self.a = None
        self.compute_primal_objective(X, y)
        
        self.res = {'w': self.w,
                    'b': self.b,
                    'time': self.t,
                    'objective': self.objective
        }
    
    #
    def svm_libsvm_solver(self, X, y, C, tol=1e-6, 
                          max_iter=100, verbose=False, gamma=0):
        N, D = X.shape
    
        if (self.kernel != 'rbf_kernel'):
            start = time.clock()
            s = svm.SVC(C=C, kernel='linear', tol=tol, max_iter=max_iter, verbose=verbose)
            s.fit(X, y)
            self.t = time.clock() - start
            self.w = s.coef_.reshape(D)
            self.b = s.intercept_
            self.status = s.fit_status_
            self.a = None
            self.support_vectors = s.support_vectors_
            
            self.compute_primal_objective(X, y)
            
            self.res = {'w': self.w,
                       'status': self.status,
                       'b': self.b,
                       'support_vectors': self.support_vectors,
                       'objective': self.objective,
                       'time': self.t
            }
        else:
            start = time.clock()
            s = svm.SVC(C=C, kernel='rbf', tol=tol, max_iter=max_iter, verbose=verbose, gamma=gamma)
            s.fit(X, y)
            self.t = time.clock() - start
            self.status = s.fit_status_
            self.a = np.zeros(N)
            self.a[s.support_] = s.dual_coef_
            self.a *= y
            self.b = s.intercept_
            self.s_vectors_X = X[s.support_]
            self.s_vectors_A = s.dual_coef_[0] * y[s.support_]
            self.s_vectors_y = y[s.support_]
            self.w = self.compute_w(X, y)
            self.compute_dual_objective()
            
            self.res = {'w': self.w,
                        'status': self.status,
                        'b': self.b,
                        'a': self.a,
                        'support_vectors': self.s_vectors_X,
                        'time': self.t,
                        'objective': self.objective
            }
            
    #Функция для подсчета опорных векторов
    def compute_support_vectors(self, X, y):
        N, D = X.shape
        
        indices_0 = [1e-6 < self.a]
        indices_1 = [self.a < self.C]
        self.indices = np.asarray((indices_0) or (indices_1)).reshape(N)
        self.s_vectors_A = self.a[self.indices]
        self.s_vectors_X = X[self.indices]
        self.s_vectors_y = y[self.indices]
   
    #Веса прямой задачи. Вызывается только в случае решения двойственной задачи с линейным ядром.
    def compute_w(self, X, y):
        if (self.kernel == 'linear_kernel'):
            self.w = np.sum((self.a * y)[:, np.newaxis] * X, axis=0)
        else:
            self.w = None
        
    #Целевая функция SVM для прямой задачи
    def compute_primal_objective(self, X, y):
        self.objective = 1/2 * (self.w).dot(self.w)
        indices = 1 - y * ((self.w).dot(X.T) + self.b) > 0
        self.objective += self.C * np.sum((1 - y * ((self.w).dot(X.T) + self.b))[indices])
        return self.objective
    
    #Целевая функция SVM для двойственной задачи
    def compute_dual_objective(self):
        self.objective = np.sum(self.s_vectors_A)
        if (self.kernel == 'linear_kernel'):
            K = linear_kernel(self.s_vectors_X, self.s_vectors_X)
        if (self.kernel == 'rbf_kernel'):
            K = rbf_kernel(self.s_vectors_X, self.s_vectors_X, gamma=self.gamma)
        P = np.outer(self.s_vectors_y, self.s_vectors_y) * K
        self.objective -= 0.5 * ((self.s_vectors_A).dot(P)).dot((self.s_vectors_A).T)
    
    #Обучение
    def fit(self, X, y):
        if (self.method == 'qp_primal'):
            self.svm_qp_primal_solver(X, y, C=self.C, tol=self.tol, 
                                      max_iter=self.max_iter, verbose=self.verbose)
        elif (self.method == 'qp_dual'):
            self.svm_qp_dual_solver(X, y, C=self.C, tol=self.tol, 
                                      max_iter=self.max_iter, 
                                    verbose=self.verbose, gamma=self.gamma)
        elif (self.method=='subgradient'):
            self.svm_subgradient_solver(X, y, C=self.C, tol=self.tol, max_iter=self.max_iter, 
                                        verbose=self.verbose, alpha=self.alpha, beta=self.beta,
                                       stochastic=self.stochastic, k=self.k)
        elif (self.method=='subgradient_1'):
            self.svm_subgradient_solver_1(X, y, C=self.C, tol=self.tol, max_iter=self.max_iter, 
                                        verbose=self.verbose, alpha=self.alpha, beta=self.beta,
                                       stochastic=self.stochastic, k=self.k)
        elif (self.method=='subgradient_2'):
            self.svm_subgradient_solver_2(X, y, C=self.C, tol=self.tol, max_iter=self.max_iter, 
                                        verbose=self.verbose, alpha=self.alpha, beta=self.beta,
                                       stochastic=self.stochastic, k=self.k)
        elif (self.method == 'liblinear'):
            self.svm_liblinear_solver(X, y, C=self.C, tol=self.tol, 
                                      max_iter=self.max_iter, verbose=self.verbose)
        elif (self.method == 'libsvm'):
            self.svm_libsvm_solver(X, y, C=self.C, tol=self.tol, 
                                      max_iter=self.max_iter, verbose=self.verbose, gamma=self.gamma)
    
    #Ответы для тестовой выборки
    def predict(self, X):
        N, D = X.shape
        if (self.w is not None):
            return np.sign(X.dot(self.w) + self.b)
        else:
            predict = np.zeros(N)
            if (self.kernel == 'linear_kernel'):
                K = linear_kernel(X, self.s_vectors_X)
                predict = np.sum((self.s_vectors_A * self.s_vectors_y)[np.newaxis, :] * K, axis=1)
            elif (self.kernel == 'rbf_kernel'):
                K = rbf_kernel(X, self.s_vectors_X, gamma=self.gamma)
                predict = np.sum((self.s_vectors_A * self.s_vectors_y)[np.newaxis, :] * K, axis=1)
            return np.sign(predict + self.b)
    
    #Визуализация объектов обучающей выборки, опорных векторов, разделяющей поверхности
    def visualize(self, X, y, rbf_kernel=None, support=False):
        N, D = X.shape
        
        if (D != 2):
            raise TypeError('X shape must be (, 2)')
        
        if ((rbf_kernel is not None) and (self.a is None)):
            raise ValueError('A must not be None')
        
        if ((self.a is not None) and (rbf_kernel is None)):
            coef = -self.w[0] / self.w[1]
            x_min, x_max = X[:, 0].min() - 2, X[:, 0].max() + 2
            xx = np.linspace(x_min, x_max)
            yy = coef * xx - self.b / self.w[1]
            if (support == True):
                s_down = self.s_vectors_X[0]
                yy_down = coef * xx + (s_down[1] - coef * s_down[0])
                s_up = self.s_vectors_X[-1]
                yy_up = coef * xx + (s_up[1] - coef * s_up[0])
            pyplot.plot(xx, yy, 'k-')
            if (support == True):
                pyplot.plot(xx, yy_down, 'k--')
                pyplot.plot(xx, yy_up, 'k--')
            pyplot.scatter(X[:, 0], X[:, 1], c=y)
            if (support == True):
                pyplot.scatter(self.s_vectors_X[:, 0], self.s_vectors_X[:, 1], s=90, facecolors='none')
            pyplot.show()
            
        if ((self.a is None) and (rbf_kernel is None)):
            coef = -self.w[0] / self.w[1]
            x_min, x_max = X[:, 0].min() - 2, X[:, 0].max() + 2
            xx = np.linspace(x_min, x_max)
            yy = coef * xx - self.b / self.w[1]
            pyplot.plot(xx, yy, 'k-')
            pyplot.scatter(X[:, 0], X[:, 1], c=y)
            pyplot.show()
        
        if ((rbf_kernel is not None) and (self.a is not None)):
            step = 0.02
            x_min, x_max = X[:, 0].min() - 2, X[:, 0].max() + 2
            y_min, y_max = X[:, 1].min() - 2, X[:, 1].max() + 2

            xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                                 np.arange(y_min, y_max, step))
            Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            pyplot.contourf(xx, yy, Z, cmap=pyplot.cm.Paired)
            pyplot.scatter(X[:, 0], X[:, 1], c=y)
            if (support == True):
                pyplot.scatter(self.s_vectors_X[:, 0], self.s_vectors_X[:, 1], s=90, facecolors='none')
            pyplot.show()
            
    def precision(self, X, y):
        y_predict = self.predict(X)
        return np.sum(y_predict == y) / len(X)