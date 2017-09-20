# Функции, необходимые для исследований

import my_svm
import numpy as np
from matplotlib import pyplot
from pylab import *
#%matplotlib inline
from importlib import reload
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold
import pandas as pd

def figure_params():
    pyplot.rcParams['figure.figsize'] = (20, 10)

#генерация данных

def generate_data(n_samples_1, n_samples_2, D):
    N = n_samples_1 + n_samples_2
    mean = np.zeros(D)
    cov = np.diag(3 * np.ones(D))
    rng = np.random.RandomState(0)
    variation = np.random.choice(20, D)
    X = np.r_[1.5 * np.random.multivariate_normal(mean, cov, n_samples_1),
              1.5 * rng.multivariate_normal(mean, cov, n_samples_2) + variation]
    y = np.asarray([-1] * (n_samples_1) + [1] * (n_samples_2))
    return (X, y)

#Первый пункт

t_qp_primal = np.zeros((10, 10))
t_qp_dual = np.zeros((10, 10))
t_subg = np.zeros((10, 10))
t_subg_stoch = np.zeros((10, 10))
t_liblinear = np.zeros((10, 10))
t_libsvm = np.zeros((10, 10))
obj_qp_primal = np.zeros((10, 10))
obj_qp_dual = np.zeros((10, 10))
obj_subg = np.zeros((10, 10))
obj_subg_stoch = np.zeros((10, 10))
obj_liblinear = np.zeros((10, 10))
obj_libsvm = np.zeros((10, 10))

n_sample = [100 * x for x in range(1, 11)]
dim = [2 * x for x in range(1, 11)]

def section_1():
    for n in n_sample:
        for d in dim:
            X, y = generate_data(n, n, d)
            qp_primal = my_svm.SVM(method='qp_primal')
            qp_primal.fit(X, y)
            t_qp_primal[n / 100 - 1, d / 2 - 1] = qp_primal.res['time']
            obj_qp_primal[n / 100 - 1, d / 2 - 1] = qp_primal.res['objective']
            qp_dual = my_svm.SVM(method='qp_dual')
            qp_dual.fit(X, y)
            t_qp_dual[n / 100 - 1, d / 2 - 1] = qp_dual.res['time']
            obj_qp_dual[n / 100 - 1, d / 2 - 1] = qp_dual.res['objective']
            subg = my_svm.SVM(method='subgradient', alpha=1e-3, tol=1e-2)
            subg.fit(X, y)
            t_subg[n / 100 - 1, d / 2 - 1] = subg.res['time']
            obj_subg[n / 100 - 1, d / 2 - 1] = subg.res['objective']
            subg_stoch = my_svm.SVM(method='subgradient', alpha=1e-3, tol=1e-2, stochastic=True, k=100)
            subg_stoch.fit(X, y)
            t_subg_stoch[n / 100 - 1, d / 2 - 1] = subg_stoch.res['time']
            obj_subg_stoch[n / 100 - 1, d / 2 - 1] = subg_stoch.res['objective']
            liblinear = my_svm.SVM(method='liblinear')
            liblinear.fit(X, y)
            t_liblinear[n / 100 - 1, d / 2 - 1] = liblinear.res['time']
            obj_liblinear[n / 100 - 1, d / 2 - 1] = liblinear.res['objective']
            libsvm = my_svm.SVM(method='libsvm')
            libsvm.fit(X, y)
            t_libsvm[n / 100 - 1, d / 2 - 1] = libsvm.res['time']
            obj_libsvm[n / 100 - 1, d / 2 - 1] = libsvm.res['objective']

def graph_1_1():
    fig = pyplot.figure()
    fig.suptitle('Number of samples: 100', fontsize=30)
    a = fig.add_subplot(1, 1, 0.5)
    pyplot.plot(dim, t_qp_primal[0], 'o-',
                dim, t_qp_dual[0], 'o-',
                dim, t_subg[0], 'o-',
                dim, t_subg_stoch[0], 'o-',
                dim, t_liblinear[0], 'o-',
                dim, t_libsvm[0], 'o-')
    a.set_yscale('log')
    pyplot.ylabel('Time, s', fontsize=20)
    pyplot.xlabel('Dimension', fontsize=20)
    pyplot.legend(['qp_primal', 'qp_dual', 'subgradient', 
               'stochastic sub', 'liblinear', 'libsvm'], bbox_to_anchor=(0.15, 1))
def graph_1_2():
    fig = pyplot.figure()
    fig.suptitle('Number of samples: 200', fontsize=30)
    a = fig.add_subplot(1, 1, 0.5)
    pyplot.plot(dim, t_qp_primal[1], 'o-',
                dim, t_qp_dual[1], 'o-',
                dim, t_subg[1], 'o-',
                dim, t_subg_stoch[1], 'o-',
                dim, t_liblinear[1], 'o-',
                dim, t_libsvm[1], 'o-')
    a.set_yscale('log')
    pyplot.ylabel('Time, s', fontsize=20)
    pyplot.xlabel('Dimension', fontsize=20)
    pyplot.legend(['qp_primal', 'qp_dual', 'subgradient', 
               'stochastic sub', 'liblinear', 'libsvm'], bbox_to_anchor=(0.15, 1))    

def graph_1_3():    
    fig = pyplot.figure()
    fig.suptitle('Number of samples: 300', fontsize=30)
    a = fig.add_subplot(1, 1, 0.5)
    pyplot.plot(dim, t_qp_primal[2], 'o-',
                dim, t_qp_dual[2], 'o-',
                dim, t_subg[2], 'o-',
                dim, t_subg_stoch[2], 'o-',
                dim, t_liblinear[2], 'o-',
                dim, t_libsvm[2], 'o-')
    a.set_yscale('log')
    pyplot.ylabel('Time, s', fontsize=20)
    pyplot.xlabel('Dimension', fontsize=20)
    pyplot.legend(['qp_primal', 'qp_dual', 'subgradient', 
               'stochastic sub', 'liblinear', 'libsvm'], bbox_to_anchor=(0.15, 1))

def graph_1_4():
    fig = pyplot.figure()
    fig.suptitle('Number of samples: 400', fontsize=30)
    a = fig.add_subplot(1, 1, 0.5)
    pyplot.plot(dim, t_qp_primal[3], 'o-',
                dim, t_qp_dual[3], 'o-',
                dim, t_subg[3], 'o-',
                dim, t_subg_stoch[3], 'o-',
                dim, t_liblinear[3], 'o-',
                dim, t_libsvm[3], 'o-')
    a.set_yscale('log')
    pyplot.ylabel('Time, s', fontsize=20)
    pyplot.xlabel('Dimension', fontsize=20)
    pyplot.legend(['qp_primal', 'qp_dual', 'subgradient', 
               'stochastic sub', 'liblinear', 'libsvm'], bbox_to_anchor=(0.15, 1))
            
def graph_1_5():
    fig = pyplot.figure()
    fig.suptitle('Number of samples: 500', fontsize=30)
    a = fig.add_subplot(1, 1, 0.5)
    pyplot.plot(dim, t_qp_primal[4], 'o-',
                dim, t_qp_dual[4], 'o-',
                dim, t_subg[4], 'o-',
                dim, t_subg_stoch[4], 'o-',
                dim, t_liblinear[4], 'o-',
                dim, t_libsvm[4], 'o-')
    a.set_yscale('log')
    pyplot.ylabel('Time, s', fontsize=20)
    pyplot.xlabel('Dimension', fontsize=20)
    pyplot.legend(['qp_primal', 'qp_dual', 'subgradient', 
               'stochastic sub', 'liblinear', 'libsvm'], bbox_to_anchor=(0.15, 1))

def graph_1_6():
    fig = pyplot.figure()
    fig.suptitle('Number of samples: 600', fontsize=30)
    a = fig.add_subplot(1, 1, 0.5)
    pyplot.plot(dim, t_qp_primal[5], 'o-',
                dim, t_qp_dual[5], 'o-',
                dim, t_subg[5], 'o-',
                dim, t_subg_stoch[5], 'o-',
                dim, t_liblinear[5], 'o-',
                dim, t_libsvm[5], 'o-')
    a.set_yscale('log')
    pyplot.ylabel('Time, s', fontsize=20)
    pyplot.xlabel('Dimension', fontsize=20)
    pyplot.legend(['qp_primal', 'qp_dual', 'subgradient', 
               'stochastic sub', 'liblinear', 'libsvm'], bbox_to_anchor=(0.15, 1))

def graph_1_7():
    fig = pyplot.figure()
    fig.suptitle('Number of samples: 700', fontsize=30)
    a = fig.add_subplot(1, 1, 0.5)
    pyplot.plot(dim, t_qp_primal[6], 'o-',
                dim, t_qp_dual[6], 'o-',
                dim, t_subg[6], 'o-',
                dim, t_subg_stoch[6], 'o-',
                dim, t_liblinear[6], 'o-',
                dim, t_libsvm[6], 'o-')
    a.set_yscale('log')
    pyplot.ylabel('Time, s', fontsize=20)
    pyplot.xlabel('Dimension', fontsize=20)
    pyplot.legend(['qp_primal', 'qp_dual', 'subgradient', 
               'stochastic sub', 'liblinear', 'libsvm'], bbox_to_anchor=(0.15, 1))
    
def graph_1_8():
    fig = pyplot.figure()
    fig.suptitle('Number of samples: 800', fontsize=30)
    a = fig.add_subplot(1, 1, 0.5)
    pyplot.plot(dim, t_qp_primal[7], 'o-',
                dim, t_qp_dual[7], 'o-',
                dim, t_subg[7], 'o-',
                dim, t_subg_stoch[7], 'o-',
                dim, t_liblinear[7], 'o-',
                dim, t_libsvm[7], 'o-')
    a.set_yscale('log')
    pyplot.ylabel('Time, s', fontsize=20)
    pyplot.xlabel('Dimension', fontsize=20)
    pyplot.legend(['qp_primal', 'qp_dual', 'subgradient', 
               'stochastic sub', 'liblinear', 'libsvm'], bbox_to_anchor=(0.15, 1))

def graph_1_9():
    fig = pyplot.figure()
    fig.suptitle('Number of samples: 900', fontsize=30)
    a = fig.add_subplot(1, 1, 0.5)
    pyplot.plot(dim, t_qp_primal[8], 'o-',
                dim, t_qp_dual[8], 'o-',
                dim, t_subg[8], 'o-',
                dim, t_subg_stoch[8], 'o-',
                dim, t_liblinear[8], 'o-',
                dim, t_libsvm[8], 'o-')
    a.set_yscale('log')
    pyplot.ylabel('Time, s', fontsize=20)
    pyplot.xlabel('Dimension', fontsize=20)
    pyplot.legend(['qp_primal', 'qp_dual', 'subgradient', 
               'stochastic sub', 'liblinear', 'libsvm'], bbox_to_anchor=(0.15, 1))

def graph_1_10():
    fig = pyplot.figure()
    fig.suptitle('Number of samples: 1000', fontsize=30)
    a = fig.add_subplot(1, 1, 0.5)
    pyplot.plot(dim, t_qp_primal[9], 'o-',
                dim, t_qp_dual[9], 'o-',
                dim, t_subg[9], 'o-',
                dim, t_subg_stoch[9], 'o-',
                dim, t_liblinear[9], 'o-',
                dim, t_libsvm[9], 'o-')
    a.set_yscale('log')
    pyplot.ylabel('Time, s', fontsize=20)
    pyplot.xlabel('Dimension', fontsize=20)
    pyplot.legend(['qp_primal', 'qp_dual', 'subgradient', 
               'stochastic sub', 'liblinear', 'libsvm'], bbox_to_anchor=(0.15, 1))
    
def graph_obj_1_1():
    fig = pyplot.figure()
    fig.suptitle('Number of samples: 100', fontsize=30)
    a = fig.add_subplot(1, 1, 0.5)
    pyplot.plot(dim, obj_qp_primal[0],
                dim, obj_qp_dual[0],
                dim, obj_subg[0], 
                dim, obj_subg_stoch[0], 
                dim, obj_liblinear[0], 
                dim, obj_libsvm[0])
    a.set_yscale('log')
    pyplot.ylabel('Objective value', fontsize=20)
    pyplot.xlabel('Dimension', fontsize=20)
    pyplot.legend(['qp_primal', 'qp_dual', 'subgradient', 
               'stochastic sub', 'liblinear', 'libsvm'], bbox_to_anchor=(0.15, 1))
    
def graph_obj_1_2():
    fig = pyplot.figure()
    fig.suptitle('Number of samples: 200', fontsize=30)
    a = fig.add_subplot(1, 1, 0.5)
    pyplot.plot(dim, obj_qp_primal[1],
                dim, obj_qp_dual[1],
                dim, obj_subg[1],
                dim, obj_subg_stoch[1],
                dim, obj_liblinear[1],
                dim, obj_libsvm[1])
    a.set_yscale('log')
    pyplot.ylabel('Objective value', fontsize=20)
    pyplot.xlabel('Dimension', fontsize=20)
    pyplot.legend(['qp_primal', 'qp_dual', 'subgradient', 
               'stochastic sub', 'liblinear', 'libsvm'], bbox_to_anchor=(0.15, 1)) 
    
def graph_obj_1_3():
    fig = pyplot.figure()
    fig.suptitle('Number of samples: 300', fontsize=30)
    a = fig.add_subplot(1, 1, 0.5)
    pyplot.plot(dim, obj_qp_primal[2],
                dim, obj_qp_dual[2],
                dim, obj_subg[2],
                dim, obj_subg_stoch[2],
                dim, obj_liblinear[2],
                dim, obj_libsvm[2])
    a.set_yscale('log')
    pyplot.ylabel('Objective value', fontsize=20)
    pyplot.xlabel('Dimension', fontsize=20)
    pyplot.legend(['qp_primal', 'qp_dual', 'subgradient', 
               'stochastic sub', 'liblinear', 'libsvm'], bbox_to_anchor=(0.15, 1))
    
def graph_obj_1_4():
    fig = pyplot.figure()
    fig.suptitle('Number of samples: 400', fontsize=30)
    a = fig.add_subplot(1, 1, 0.5)
    pyplot.plot(dim, obj_qp_primal[3],
                dim, obj_qp_dual[3],
                dim, obj_subg[3],
                dim, obj_subg_stoch[3],
                dim, obj_liblinear[3],
                dim, obj_libsvm[3])
    a.set_yscale('log')
    pyplot.ylabel('Objective value', fontsize=20)
    pyplot.xlabel('Dimension', fontsize=20)
    pyplot.legend(['qp_primal', 'qp_dual', 'subgradient', 
               'stochastic sub', 'liblinear', 'libsvm'], bbox_to_anchor=(0.15, 1))
    
def graph_obj_1_5():
    fig = pyplot.figure()
    fig.suptitle('Number of samples: 500', fontsize=30)
    a = fig.add_subplot(1, 1, 0.5)
    pyplot.plot(dim, obj_qp_primal[4], 
                dim, obj_qp_dual[4], 
                dim, obj_subg[4], 
                dim, obj_subg_stoch[4], 
                dim, obj_liblinear[4],
                dim, obj_libsvm[4]) 
    a.set_yscale('log')
    pyplot.ylabel('Objective value', fontsize=20)
    pyplot.xlabel('Dimension', fontsize=20)
    pyplot.legend(['qp_primal', 'qp_dual', 'subgradient', 
               'stochastic sub', 'liblinear', 'libsvm'], bbox_to_anchor=(0.15, 1))
    
def graph_obj_1_6():
    fig = pyplot.figure()
    fig.suptitle('Number of samples: 600', fontsize=30)
    a = fig.add_subplot(1, 1, 0.5)
    pyplot.plot(dim, obj_qp_primal[5],
                dim, obj_qp_dual[5],
                dim, obj_subg[5],
                dim, obj_subg_stoch[5],
                dim, obj_liblinear[5],
                dim, obj_libsvm[5])
    a.set_yscale('log')
    pyplot.ylabel('Objective value', fontsize=20)
    pyplot.xlabel('Dimension', fontsize=20)
    pyplot.legend(['qp_primal', 'qp_dual', 'subgradient', 
               'stochastic sub', 'liblinear', 'libsvm'], bbox_to_anchor=(0.15, 1))
def graph_obj_1_7():
    fig = pyplot.figure()
    fig.suptitle('Number of samples: 700', fontsize=30)
    a = fig.add_subplot(1, 1, 0.5)
    pyplot.plot(dim, obj_qp_primal[6],
                dim, obj_qp_dual[6],
                dim, obj_subg[6],
                dim, obj_subg_stoch[6],
                dim, obj_liblinear[6],
                dim, obj_libsvm[6])
    a.set_yscale('log')
    pyplot.ylabel('Objective value', fontsize=20)
    pyplot.xlabel('Dimension', fontsize=20)
    pyplot.legend(['qp_primal', 'qp_dual', 'subgradient', 
               'stochastic sub', 'liblinear', 'libsvm'], bbox_to_anchor=(0.15, 1))
    
def graph_obj_1_8():
    fig = pyplot.figure()
    fig.suptitle('Number of samples: 800', fontsize=30)
    a = fig.add_subplot(1, 1, 0.5)
    pyplot.plot(dim, obj_qp_primal[7],
                dim, obj_qp_dual[7],
                dim, obj_subg[7],
                dim, obj_subg_stoch[7],
                dim, obj_liblinear[7],
                dim, obj_libsvm[7])
    a.set_yscale('log')
    pyplot.ylabel('Objective value', fontsize=20)
    pyplot.xlabel('Dimension', fontsize=20)
    pyplot.legend(['qp_primal', 'qp_dual', 'subgradient', 
               'stochastic sub', 'liblinear', 'libsvm'], bbox_to_anchor=(0.15, 1))
    
def graph_obj_1_9():
    fig = pyplot.figure()
    fig.suptitle('Number of samples: 900', fontsize=30)
    a = fig.add_subplot(1, 1, 0.5)
    pyplot.plot(dim, obj_qp_primal[8],
                dim, obj_qp_dual[8],
                dim, obj_subg[8],
                dim, obj_subg_stoch[8],
                dim, obj_liblinear[8],
                dim, obj_libsvm[8])
    a.set_yscale('log')
    pyplot.ylabel('Objective value', fontsize=20)
    pyplot.xlabel('Dimension', fontsize=20)
    pyplot.legend(['qp_primal', 'qp_dual', 'subgradient', 
               'stochastic sub', 'liblinear', 'libsvm'], bbox_to_anchor=(0.15, 1))
    
def graph_obj_1_10():
    fig = pyplot.figure()
    fig.suptitle('Number of samples: 1000', fontsize=30)
    a = fig.add_subplot(1, 1, 0.5)
    pyplot.plot(dim, obj_qp_primal[9],
                dim, obj_qp_dual[9],
                dim, obj_subg[9],
                dim, obj_subg_stoch[9],
                dim, obj_liblinear[9],
                dim, obj_libsvm[9])
    a.set_yscale('log')
    pyplot.ylabel('Objective value', fontsize=20)
    pyplot.xlabel('Dimension', fontsize=20)
    pyplot.legend(['qp_primal', 'qp_dual', 'subgradient', 
               'stochastic sub', 'liblinear', 'libsvm'], bbox_to_anchor=(0.15, 1))
    
#Второй пункт

n_sample = [100 * x for x in range(1, 11)]
dim = [2 * x for x in range(1, 11)]            
rbf_t_qp_dual = np.zeros((10, 10))
rbf_t_libsvm = np.zeros((10, 10))
rbf_obj_qp_dual = np.zeros((10, 10))
rbf_obj_libsvm = np.zeros((10, 10))

def section_2():
    for n in n_sample:
        for d in dim:
            X, y = generate_data(n, n, d)
            qp_dual = my_svm.SVM(method='qp_dual', kernel='rbf_kernel', gamma=0.01)
            qp_dual.fit(X, y)
            rbf_t_qp_dual[n / 100 - 1, d / 2 - 1] = qp_dual.res['time']
            rbf_obj_qp_dual[n / 100 - 1, d / 2 - 1] = qp_dual.res['objective']
            libsvm = my_svm.SVM(method='libsvm', kernel='rbf_kernel', gamma=0.01)
            libsvm.fit(X, y)
            rbf_t_libsvm[n / 100 - 1, d / 2 - 1] = libsvm.res['time']
            rbf_obj_libsvm[n / 100 - 1, d / 2 - 1] = libsvm.res['objective']

def graph_2_1():
    fig = pyplot.figure()
    fig.suptitle('Number of samples: 100', fontsize=30)
    a = fig.add_subplot(1, 1, 0.5)
    pyplot.plot(dim, rbf_t_qp_dual[0], 'o-',
                dim, rbf_t_libsvm[0], 'o-')
    a.set_yscale('log')
    pyplot.ylabel('Time, s', fontsize=20)
    pyplot.xlabel('Dimension', fontsize=20)
    pyplot.legend(['qp_dual', 'libsvm'], bbox_to_anchor=(0.15, 1))

def graph_2_2():
    fig = pyplot.figure()
    fig.suptitle('Number of samples: 200', fontsize=30)
    a = fig.add_subplot(1, 1, 0.5)
    pyplot.plot(dim, rbf_t_qp_dual[1], 'o-',
                dim, rbf_t_libsvm[1], 'o-')
    a.set_yscale('log')
    pyplot.ylabel('Time, s', fontsize=20)
    pyplot.xlabel('Dimension', fontsize=20)
    pyplot.legend(['qp_dual', 'libsvm'], bbox_to_anchor=(0.15, 1))    

def graph_2_3():    
    fig = pyplot.figure()
    fig.suptitle('Number of samples: 300', fontsize=30)
    a = fig.add_subplot(1, 1, 0.5)
    pyplot.plot(dim, rbf_t_qp_dual[2], 'o-',
                dim, rbf_t_libsvm[2], 'o-')
    a.set_yscale('log')
    pyplot.ylabel('Time, s', fontsize=20)
    pyplot.xlabel('Dimension', fontsize=20)
    pyplot.legend(['qp_dual', 'libsvm'], bbox_to_anchor=(0.15, 1))

def graph_2_4():
    fig = pyplot.figure()
    fig.suptitle('Number of samples: 400', fontsize=30)
    a = fig.add_subplot(1, 1, 0.5)
    pyplot.plot(dim, rbf_t_qp_dual[3], 'o-',
                dim, rbf_t_libsvm[3], 'o-')
    a.set_yscale('log')
    pyplot.ylabel('Time, s', fontsize=20)
    pyplot.xlabel('Dimension', fontsize=20)
    pyplot.legend(['qp_dual', 'libsvm'], bbox_to_anchor=(0.15, 1))
            
def graph_2_5():
    fig = pyplot.figure()
    fig.suptitle('Number of samples: 500', fontsize=30)
    a = fig.add_subplot(1, 1, 0.5)
    pyplot.plot(dim, rbf_t_qp_dual[4], 'o-',
                dim, rbf_t_libsvm[4], 'o-')
    a.set_yscale('log')
    pyplot.ylabel('Time, s', fontsize=20)
    pyplot.xlabel('Dimension', fontsize=20)
    pyplot.legend(['qp_dual', 'libsvm'], bbox_to_anchor=(0.15, 1))

def graph_2_6():
    fig = pyplot.figure()
    fig.suptitle('Number of samples: 600', fontsize=30)
    a = fig.add_subplot(1, 1, 0.5)
    pyplot.plot(dim, rbf_t_qp_dual[5], 'o-',
                dim, rbf_t_libsvm[5], 'o-')
    a.set_yscale('log')
    pyplot.ylabel('Time, s', fontsize=20)
    pyplot.xlabel('Dimension', fontsize=20)
    pyplot.legend(['qp_dual', 'libsvm'], bbox_to_anchor=(0.15, 1))

def graph_2_7():
    fig = pyplot.figure()
    fig.suptitle('Number of samples: 700', fontsize=30)
    a = fig.add_subplot(1, 1, 0.5)
    pyplot.plot(dim, rbf_t_qp_dual[6], 'o-',
                dim, rbf_t_libsvm[6], 'o-')
    a.set_yscale('log')
    pyplot.ylabel('Time, s', fontsize=20)
    pyplot.xlabel('Dimension', fontsize=20)
    pyplot.legend(['qp_dual', 'libsvm'], bbox_to_anchor=(0.15, 1))
    
def graph_2_8():
    fig = pyplot.figure()
    fig.suptitle('Number of samples: 800', fontsize=30)
    a = fig.add_subplot(1, 1, 0.5)
    pyplot.plot(dim, rbf_t_qp_dual[7], 'o-',
                dim, rbf_t_libsvm[7], 'o-')
    a.set_yscale('log')
    pyplot.ylabel('Time, s', fontsize=20)
    pyplot.xlabel('Dimension', fontsize=20)
    pyplot.legend(['qp_dual', 'libsvm'], bbox_to_anchor=(0.15, 1))

def graph_2_9():
    fig = pyplot.figure()
    fig.suptitle('Number of samples: 900', fontsize=30)
    a = fig.add_subplot(1, 1, 0.5)
    pyplot.plot(dim, rbf_t_qp_dual[8], 'o-',
                dim, rbf_t_libsvm[8], 'o-')
    a.set_yscale('log')
    pyplot.ylabel('Time, s', fontsize=20)
    pyplot.xlabel('Dimension', fontsize=20)
    pyplot.legend(['qp_dual', 'libsvm'], bbox_to_anchor=(0.15, 1))

def graph_2_10():
    fig = pyplot.figure()
    fig.suptitle('Number of samples: 1000', fontsize=30)
    a = fig.add_subplot(1, 1, 0.5)
    pyplot.plot(dim, rbf_t_qp_dual[9], 'o-',
                dim, rbf_t_libsvm[9], 'o-')
    a.set_yscale('log')
    pyplot.ylabel('Time, s', fontsize=20)
    pyplot.xlabel('Dimension', fontsize=20)
    pyplot.legend(['qp_dual', 'libsvm'], bbox_to_anchor=(0.15, 1))

def graph_obj_2_1():
    fig = pyplot.figure()
    fig.suptitle('Number of samples: 100', fontsize=30)
    a = fig.add_subplot(1, 1, 0.5)
    pyplot.plot(dim, rbf_obj_qp_dual[0],
                dim, rbf_obj_libsvm[0])
    a.set_yscale('log')
    pyplot.ylabel('Objective value', fontsize=20)
    pyplot.xlabel('Dimension', fontsize=20)
    pyplot.legend(['qp_dual', 'libsvm'], bbox_to_anchor=(0.15, 1))
    
def graph_obj_2_2():
    fig = pyplot.figure()
    fig.suptitle('Number of samples: 200', fontsize=30)
    a = fig.add_subplot(1, 1, 0.5)
    pyplot.plot(dim, rbf_obj_qp_dual[1],
                dim, rbf_obj_libsvm[1])
    a.set_yscale('log')
    pyplot.ylabel('Objective value', fontsize=20)
    pyplot.xlabel('Dimension', fontsize=20)
    pyplot.legend(['qp_dual', 'libsvm'], bbox_to_anchor=(0.15, 1)) 
    
def graph_obj_2_3():
    fig = pyplot.figure()
    fig.suptitle('Number of samples: 300', fontsize=30)
    a = fig.add_subplot(1, 1, 0.5)
    pyplot.plot(dim, rbf_obj_qp_dual[2],
                dim, rbf_obj_libsvm[2])
    a.set_yscale('log')
    pyplot.ylabel('Objective value', fontsize=20)
    pyplot.xlabel('Dimension', fontsize=20)
    pyplot.legend(['qp_dual', 'libsvm'], bbox_to_anchor=(0.15, 1))
    
def graph_obj_2_4():
    fig = pyplot.figure()
    fig.suptitle('Number of samples: 400', fontsize=30)
    a = fig.add_subplot(1, 1, 0.5)
    pyplot.plot(dim, rbf_obj_qp_dual[3],
                dim, rbf_obj_libsvm[3])
    a.set_yscale('log')
    pyplot.ylabel('Objective value', fontsize=20)
    pyplot.xlabel('Dimension', fontsize=20)
    pyplot.legend(['qp_dual', 'libsvm'], bbox_to_anchor=(0.15, 1))
    
def graph_obj_2_5():
    fig = pyplot.figure()
    fig.suptitle('Number of samples: 500', fontsize=30)
    a = fig.add_subplot(1, 1, 0.5)
    pyplot.plot(dim, rbf_obj_qp_dual[4], 
                dim, rbf_obj_libsvm[4]) 
    a.set_yscale('log')
    pyplot.ylabel('Objective value', fontsize=20)
    pyplot.xlabel('Dimension', fontsize=20)
    pyplot.legend(['qp_dual', 'libsvm'], bbox_to_anchor=(0.15, 1))
    
def graph_obj_2_6():
    fig = pyplot.figure()
    fig.suptitle('Number of samples: 600', fontsize=30)
    a = fig.add_subplot(1, 1, 0.5)
    pyplot.plot(dim, rbf_obj_qp_dual[5],
                dim, rbf_obj_libsvm[5])
    a.set_yscale('log')
    pyplot.ylabel('Objective value', fontsize=20)
    pyplot.xlabel('Dimension', fontsize=20)
    pyplot.legend(['qp_dual', 'libsvm'], bbox_to_anchor=(0.15, 1))

def graph_obj_2_7():
    fig = pyplot.figure()
    fig.suptitle('Number of samples: 700', fontsize=30)
    a = fig.add_subplot(1, 1, 0.5)
    pyplot.plot(dim, rbf_obj_qp_dual[6],
                dim, rbf_obj_libsvm[6])
    a.set_yscale('log')
    pyplot.ylabel('Objective value', fontsize=20)
    pyplot.xlabel('Dimension', fontsize=20)
    pyplot.legend(['qp_dual', 'libsvm'], bbox_to_anchor=(0.15, 1))
    
def graph_obj_2_8():
    fig = pyplot.figure()
    fig.suptitle('Number of samples: 800', fontsize=30)
    a = fig.add_subplot(1, 1, 0.5)
    pyplot.plot(dim, rbf_obj_qp_dual[7],
                dim, rbf_obj_libsvm[7])
    a.set_yscale('log')
    pyplot.ylabel('Objective value', fontsize=20)
    pyplot.xlabel('Dimension', fontsize=20)
    pyplot.legend(['qp_dual', 'libsvm'], bbox_to_anchor=(0.15, 1))
    
def graph_obj_2_9():
    fig = pyplot.figure()
    fig.suptitle('Number of samples: 900', fontsize=30)
    a = fig.add_subplot(1, 1, 0.5)
    pyplot.plot(dim, rbf_obj_qp_dual[8],
                dim, rbf_obj_libsvm[8])
    a.set_yscale('log')
    pyplot.ylabel('Objective value', fontsize=20)
    pyplot.xlabel('Dimension', fontsize=20)
    pyplot.legend(['qp_dual', 'libsvm'], bbox_to_anchor=(0.15, 1))
    
def graph_obj_2_10():
    fig = pyplot.figure()
    fig.suptitle('Number of samples: 1000', fontsize=30)
    a = fig.add_subplot(1, 1, 0.5)
    pyplot.plot(dim, rbf_obj_qp_dual[9],
                dim, rbf_obj_libsvm[9])
    a.set_yscale('log')
    pyplot.ylabel('Objective value', fontsize=20)
    pyplot.xlabel('Dimension', fontsize=20)
    pyplot.legend(['qp_dual', 'libsvm'], bbox_to_anchor=(0.15, 1))

#Третий пункт

#Линейно разделимая выборка
def generate_great_data(n_samples_1, n_samples_2, D):
    N = n_samples_1 + n_samples_2
    mean = np.zeros(D)
    cov = np.diag(3 * np.ones(D))
    rng = np.random.RandomState(0)
    variation = np.ones(D) * 20
    X = np.r_[1.5 * np.random.multivariate_normal(mean, cov, n_samples_1),
              1.5 * rng.multivariate_normal(mean, cov, n_samples_2) + variation]
    y = np.asarray([-1] * (n_samples_1) + [1] * (n_samples_2))
    return (X, y)

#Трудно разделимая выборка
def generate_bad_data(n_samples_1, n_samples_2, D):
    N = n_samples_1 + n_samples_2
    mean = np.zeros(D)
    cov = np.diag(3 * np.ones(D))
    rng = np.random.RandomState(0)
    variation = np.random.choice(np.arange(1, 10), D)
    X = np.r_[1.5 * np.random.multivariate_normal(mean, cov, n_samples_1),
              1.5 * rng.multivariate_normal(mean, cov, n_samples_2) + variation]
    y = np.asarray([-1] * (n_samples_1) + [1] * (n_samples_2))
    return (X, y)

#кросс-валидация
def cross_val_C_great():
    X_great, y_great = generate_great_data(2000, 2000, 2)
    C_vals = np.linspace(1, 100, 100)
    quality_vals_C_great = np.array([])
    
    for c in C_vals:
        alg = my_svm.SVM(kernel='linear_kernel', C=c)
        qual = []
        cross_v = KFold(n=len(X_great), n_folds=5)
        for train_idx, test_idx in cross_v:
            alg.fit(X_great[train_idx], y_great[train_idx])
            qual += [alg.precision(X_great[test_idx], y_great[test_idx])]
        qual = np.asarray(qual)
        quality_vals_C_great = np.append(quality_vals_C_great, qual.mean())
    fig = pyplot.figure()
    fig.suptitle('Linear separable', fontsize=30)
    pyplot.plot(C_vals, quality_vals_C_great)
    pyplot.xlabel("C vals", fontsize=20)
    pyplot.ylabel("Precision", fontsize=20)
    pyplot.show()
    return C_vals[np.argmax(quality_vals_C_great)]

def cross_val_C_bad():
    X_bad, y_bad = generate_bad_data(2000, 2000, 2)
    C_vals = np.linspace(1, 100, 100)
    quality_vals_C_bad = np.array([])
    
    for c in C_vals:
        alg = my_svm.SVM(kernel='linear_kernel', C=c)
        qual = []
        cross_v = KFold(n=len(X_bad), n_folds=5)
        for train_idx, test_idx in cross_v:
            alg.fit(X_bad[train_idx], y_bad[train_idx])
            qual += [alg.precision(X_bad[test_idx], y_bad[test_idx])]
        qual = np.asarray(qual)
        quality_vals_C_bad = np.append(quality_vals_C_bad, qual.mean())
    fig = pyplot.figure()
    fig.suptitle('Bad separable', fontsize=30)
    pyplot.plot(C_vals, quality_vals_C_bad)
    pyplot.xlabel("C vals", fontsize=20)
    pyplot.ylabel("Precision", fontsize=20)
    pyplot.show()
    return C_vals[np.argmax(quality_vals_C_bad)]

def cross_val_gamma_great():
    X_great, y_great = generate_great_data(2000, 2000, 2)
    gamma_vals = np.linspace(0.001, 1, 200)
    quality_vals_gamma_great = np.array([])
    
    for gamma in gamma_vals:
        alg = my_svm.SVM(kernel='rbf_kernel', gamma=gamma)
        qual = []
        cross_v = KFold(n=len(X_great), n_folds=5)
        for train_idx, test_idx in cross_v:
            alg.fit(X_great[train_idx], y_great[train_idx])
            qual += [alg.precision(X_great[test_idx], y_great[test_idx])]
        qual = np.asarray(qual)
        quality_vals_gamma_great = np.append(quality_vals_gamma_great, qual.mean())
    fig = pyplot.figure()
    fig.suptitle('Linear separable', fontsize=30)
    pyplot.plot(gamma_vals, quality_vals_gamma_great)
    pyplot.xlabel("Gamma vals", fontsize=20)
    pyplot.ylabel("Precision", fontsize=20)
    pyplot.show()
    return gamma_vals[np.argmax(quality_vals_gamma_great)]

def cross_val_gamma_bad():
    X_bad, y_bad = generate_bad_data(2000, 2000, 2)
    gamma_vals = np.linspace(0.001, 1, 200)
    quality_vals_gamma_bad = np.array([])
    
    for gamma in gamma_vals:
        alg = my_svm.SVM(kernel='rbf_kernel', gamma=gamma)
        qual = []
        cross_v = KFold(n=len(X_bad), n_folds=5)
        for train_idx, test_idx in cross_v:
            alg.fit(X_bad[train_idx], y_bad[train_idx])
            qual += [alg.precision(X_bad[test_idx], y_bad[test_idx])]
        qual = np.asarray(qual)
        quality_vals_gamma_bad = np.append(quality_vals_gamma_bad, qual.mean())
    fig = pyplot.figure()
    fig.suptitle('Bad separable', fontsize=30)
    pyplot.plot(gamma_vals, quality_vals_gamma_bad)
    pyplot.xlabel("Gamma vals", fontsize=20)
    pyplot.ylabel("Precision", fontsize=20)
    pyplot.show()
    return gamma_vals[np.argmax(quality_vals_gamma_bad)]

def cross_val_C_gamma_great():
    X_great, y_great = generate_great_data(2000, 2000, 2)
    C_vals = np.linspace(1, 100, 100)
    quality_vals_C_great = np.array([])
    
    for c in C_vals:
        alg = my_svm.SVM(kernel='rbf_kernel', gamma=0.001, C=c)
        qual = []
        cross_v = KFold(n=len(X_great), n_folds=5)
        for train_idx, test_idx in cross_v:
            alg.fit(X_great[train_idx], y_great[train_idx])
            qual += [alg.precision(X_great[test_idx], y_great[test_idx])]
        qual = np.asarray(qual)
        quality_vals_C_great = np.append(quality_vals_C_great, qual.mean())
    fig = pyplot.figure()
    fig.suptitle('Linear separable, gamma = 0.001', fontsize=30)
    pyplot.plot(C_vals, quality_vals_C_great)
    pyplot.xlabel("C vals", fontsize=20)
    pyplot.ylabel("Precision", fontsize=20)
    pyplot.show()
    return C_vals[np.argmax(quality_vals_C_great)]

def cross_val_C_gamma_bad():
    X_bad, y_bad = generate_bad_data(2000, 2000, 2)
    C_vals = np.linspace(1, 100, 100)
    quality_vals_C_bad = np.array([])
    
    for c in C_vals:
        alg = my_svm.SVM(kernel='rbf_kernel', gamma=0.136, C=c)
        qual = []
        cross_v = KFold(n=len(X_bad), n_folds=5)
        for train_idx, test_idx in cross_v:
            alg.fit(X_bad[train_idx], y_bad[train_idx])
            qual += [alg.precision(X_bad[test_idx], y_bad[test_idx])]
        qual = np.asarray(qual)
        quality_vals_C_bad = np.append(quality_vals_C_bad, qual.mean())
    fig = pyplot.figure()
    fig.suptitle('Bad separable, gamma=0.136', fontsize=30)
    pyplot.plot(C_vals, quality_vals_C_bad)
    pyplot.xlabel("C vals", fontsize=20)
    pyplot.ylabel("Precision", fontsize=20)
    pyplot.show()
    return C_vals[np.argmax(quality_vals_C_bad)]

#Четвертый пункт

def alpha_graph(X, y):
    alpha_scale = np.linspace(0.0001, 0.01, 100)
    t_subg = []
    
    for alpha in alpha_scale:
        alg = my_svm.SVM(method='subgradient_1', alpha=alpha, max_iter=2000)
        alg.fit(X, y)
        t_subg += [alg.res['time']]
    fig = pyplot.figure()
    fig.suptitle('Subgradient with alpha', fontsize=30)
    pyplot.plot(alpha_scale, t_subg)
    pyplot.ylabel('Time, s', fontsize=20)
    pyplot.xlabel('Alpha', fontsize=20)
    pyplot.show()
    t_subg = np.asarray(t_subg)
    return (alpha_scale[np.argmin(t_subg)], alpha_scale[np.argmax(t_subg)])

def alpha_obj_graph(X, y, alpha1, alpha2):
    
    alg1 = my_svm.SVM(method='subgradient_1', alpha=alpha1, max_iter=2000)
    alg1.fit(X, y)
    alg2 = my_svm.SVM(method='qp_primal', max_iter=2000)
    alg2.fit(X, y)
    alg3 = my_svm.SVM(method='subgradient_1', alpha=alpha2, max_iter=2000)
    alg3.fit(X, y)
    x_max = min(len(alg1.res['objective_curve']), len(alg3.res['objective_curve']))
    true_obj = np.ones(x_max) * alg2.res['objective']
    fig = pyplot.figure()
    fig.suptitle('Subgradient with alpha', fontsize=30)
    a = fig.add_subplot(1, 1, 0.5)
    a.set_yscale('log')
    pyplot.plot(np.arange(x_max), alg1.res['objective_curve'][0 :x_max],
               np.arange(x_max), true_obj,
               np.arange(x_max), alg3.res['objective_curve'][0 :x_max])
    pyplot.ylabel('Objective value', fontsize=20)
    pyplot.xlabel('Iteration', fontsize=20)
    pyplot.legend(['Subgradient with the best alpha', 'True value', 
                   'Subgradient with the worst alpha'])
    pyplot.show()
    
def alpha_t_graph(X, y):
    alpha_scale = np.linspace(1e-5, 1e-4, 100)
    t_subg = []
    
    for alpha in alpha_scale:
        alg = my_svm.SVM(method='subgradient_1', alpha=alpha, beta=1.0, max_iter=2000)
        alg.fit(X, y)
        t_subg += [alg.res['time']]
    fig = pyplot.figure()
    fig.suptitle('Subgradient with alpha', fontsize=30)
    pyplot.plot(alpha_scale, t_subg)
    pyplot.ylabel('Time, s', fontsize=20)
    pyplot.xlabel('Alpha', fontsize=20)
    pyplot.show()
    t_subg = np.asarray(t_subg)
    return (alpha_scale[np.argmin(t_subg)], alpha_scale[np.argmax(t_subg)])

def alpha_t_obj_graph(X, y, alpha1, alpha2):
    
    alg1 = my_svm.SVM(method='subgradient_1', alpha=alpha1, beta=1.0, max_iter=2000)
    alg1.fit(X, y)
    alg2 = my_svm.SVM(method='qp_primal', max_iter=2000)
    alg2.fit(X, y)
    alg3 = my_svm.SVM(method='subgradient_1', alpha=alpha2, max_iter=2000)
    alg3.fit(X, y)
    x_max = min(len(alg1.res['objective_curve']), len(alg3.res['objective_curve']))
    true_obj = np.ones(x_max) * alg2.res['objective']
    fig = pyplot.figure()
    fig.suptitle('Subgradient with alpha', fontsize=30)
    a = fig.add_subplot(1, 1, 0.5)
    a.set_yscale('log')
    pyplot.plot(np.arange(x_max), alg1.res['objective_curve'][0 :x_max],
               np.arange(x_max), true_obj,
               np.arange(x_max), alg3.res['objective_curve'][0 :x_max])
    pyplot.ylabel('Objective value', fontsize=20)
    pyplot.xlabel('Iteration', fontsize=20)
    pyplot.legend(['Subgradient with the best alpha', 'True value', 
                   'Subgradient with the worst alpha'])
    pyplot.show()


def alpha_beta_graph(X, y, alpha):
    beta_scale = np.linspace(1e-6, 1e-5, 50)
    t_subg = []
    for beta in beta_scale:
        alg = my_svm.SVM(method='subgradient_1', alpha=alpha, beta=beta, max_iter=2000, tol=1e-6)
        alg.fit(X, y)
        t_subg += [alg.res['time']]
    fig = pyplot.figure()
    fig.suptitle('Subgradient with alpha and beta', fontsize=30)
    pyplot.plot(beta_scale, t_subg)
    pyplot.ylabel('Time, s', fontsize=20)
    pyplot.xlabel('Beta', fontsize=20)
    pyplot.show()
    t_subg = np.asarray(t_subg)
    return (beta_scale[np.argmin(t_subg)], beta_scale[np.argmax(t_subg)])

def alpha_beta_obj(X, y, alpha, beta1, beta2):
    alg1 = my_svm.SVM(method='subgradient_1', alpha=alpha, beta=beta1, max_iter=2000, tol=1e-6)
    alg1.fit(X, y)
    alg2 = my_svm.SVM(method='qp_primal', max_iter=2000)
    alg2.fit(X, y)
    alg3 = my_svm.SVM(method='subgradient_1', alpha=alpha, beta=beta2, max_iter=2000, tol=1e-6)
    alg3.fit(X, y)
    x_max = min(len(alg1.res['objective_curve']), len(alg3.res['objective_curve']))
    true_obj = np.ones(x_max) * alg2.res['objective']
    fig = pyplot.figure()
    fig.suptitle('Subgradient with alpha and beta', fontsize=30)
    a = fig.add_subplot(1, 1, 0.5)
    a.set_yscale('log')
    pyplot.plot(np.arange(x_max), alg1.res['objective_curve'][0 :x_max],
               np.arange(x_max), true_obj,
               np.arange(x_max), alg3.res['objective_curve'][0 :x_max])
    pyplot.ylabel('Objective value', fontsize=20)
    pyplot.xlabel('Iteration', fontsize=20)
    pyplot.legend(['Subgradient with the best beta', 'True value',
                  'Subgradient with the worst beta'])
    pyplot.show()
    
#Пятый пункт
def k_graph(X, y):
    N, D = X.shape
    
    k_scale = np.linspace(0.1, 1, 10)
    t_subg = []
    
    for k in k_scale:
        subsample = int(k * N)
        alg = my_svm.SVM(method='subgradient_1', alpha=1e-2, 
                         max_iter=3000, tol=1e-1, stochastic=True, k=subsample)
        alg.fit(X, y)
        t_subg += [alg.res['time']]
    fig = pyplot.figure()
    fig.suptitle('Stochastic subgradient with different subsample', fontsize=30)
    pyplot.plot(k_scale, t_subg)
    pyplot.ylabel('Time, s', fontsize=20)
    pyplot.xlabel('Size of subsample, portion', fontsize=20)
    pyplot.show()
    t_subg = np.asarray(t_subg)
    return (k_scale[np.argmin(t_subg)], k_scale[np.argmax(t_subg)])

def k_obj(X, y):
    N, D = X.shape
    
    k_scale = np.linspace(0.1, 1, 10)
    t_subg = []
    alg2 = my_svm.SVM(method='qp_primal', max_iter=3000)
    alg2.fit(X, y)
    true_obj = np.ones(3000) * alg2.res['objective']
    fig = pyplot.figure()
    a = fig.add_subplot(1, 1, 0.5)
    a.set_yscale('log')
        
    for k in k_scale:
        subsample = int(k * N)
        alg1 = my_svm.SVM(method='subgradient_1', alpha=1e-2, 
                         max_iter=3000, tol=1e-1, stochastic=True, k=subsample)
        alg1.fit(X, y)
        x_max = len(alg1.res['objective_curve'])
        pyplot.plot(np.arange(x_max), alg1.res['objective_curve'])
    pyplot.plot(np.arange(3000), true_obj)
    pyplot.ylabel('Objective value', fontsize=20)
    pyplot.xlabel('Iteration', fontsize=20)
    k_scale = list(k_scale)
    pyplot.legend(k_scale + ['true value'])
    pyplot.show()
    
#Шестой пункт

def visualize_libsvm(X, y):
    fig = pyplot.figure()
    fig.suptitle('LibSVM', fontsize=30)
    alg = my_svm.SVM(kernel='linear_kernel')
    alg.fit(X, y)
    alg.visualize(X, y)

def visualize_qp_dual(X, y):
    fig = pyplot.figure()
    fig.suptitle('QP dual with support vectors', fontsize=30)    
    alg = my_svm.SVM(kernel='linear_kernel', method='qp_dual')
    alg.fit(X, y)
    alg.visualize(X, y, support=True)

def visualize_libsvm_rbf(X, y):
    fig = pyplot.figure()
    fig.suptitle('LibSVM with RBF-kernel', fontsize=30)
    alg = my_svm.SVM(kernel='rbf_kernel', gamma=0.02)
    alg.fit(X, y)
    alg.visualize(X, y, rbf_kernel=True)

def visualize_subg(X, y):
    fig = pyplot.figure()
    fig.suptitle('Subgradient descent', fontsize=30)
    alg = my_svm.SVM(kernel='libear_kernel', method='subgradient', alpha=1e-3, tol=1e-3)
    alg.fit(X, y)
    alg.visualize(X, y)

def visualize_subg_stoch(X, y):
    fig = pyplot.figure()
    fig.suptitle('Stochastic subgradient descent', fontsize=30) 
    alg = my_svm.SVM(kernel='libear_kernel', method='subgradient', stochastic=True, alpha=1e-3, tol=1e-3, k=100)
    alg.fit(X, y)
    alg.visualize(X, y)    