from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import math
from pylab import *


def norm(x):
    nrm =0
    for i in rang(len(x)):
        nrm += pow(x[i], 2)
    nrm = sqrt(nrm)
    return nrm
"
def partial_derivative(func, var=0, point=[]):
    args = point[:]
    def wraps(x):
        args[var] = x
        return func(*args)
    return misc.derivative(wraps, point[var], dx = 1e-6)

 
def gradient_descent(alpha, x, y, ep=0.0001, max_iter=10000):
    converged = False
   iter = 0
    m = x.shape[0] 

    # initial theta
    t0 = np.random.random(x.shape[1])
    t1 = np.random.random(x.shape[1])

    # total error, J(theta)
    J = sum([(t0 + t1*x[i] - y[i])**2 for i in range(m)])

    # Iterate Loop
    while not converged:
        # for each training sample, compute the gradient (d/d_theta j(theta))
        grad0 = 1.0/m * sum([(t0 + t1*x[i] - y[i]) for i in range(m)]) 
        grad1 = 1.0/m * sum([(t0 + t1*x[i] - y[i])*x[i] for i in range(m)])

        # update the theta_temp
        temp0 = t0 - alpha * grad0
        temp1 = t1 - alpha * grad1
    
        # update theta
        t0 = temp0
        t1 = temp1

        # mean squared error
        e = sum( [ (t0 + t1*x[i] - y[i])**2 for i in range(m)] ) 

        if abs(J-e) <= ep:
            print 'Converged, iterations: ', iter, '!!!'
            converged = True
    
        J = e   # update error 
        iter += 1  # update iter
    
        if iter == max_iter:
            print 'Max interactions exceeded!'
            converged = True

    return t0,t1

"
########################################################################################################################################

def laplacien(g, x, Lambda):
    L = ones(len(x))
    for r in range(len(x)):
        L[r] = 2*(x[r] - g(x[r])) - Lambda*((x[r]-x[r-1])/abs(x[r]-x[r-1]))
    return L

def Ista(beta_1, beta_2, x_0, Lambda, n, ST):
    x_i = ones(len(x_0)) 
    x_i_1 = ones(len(x_0))
    z_i = ones(len(x_0))
    for i in range(len(x_0)):
        x_i[i] = x_0[i]
    for i in range(n):
        L = laplacien(g, x_i, Lambda)
        x_i_1= ST(z_i, beta_2)
        for j in range(len(x_0)):
            z_i[j] = x_i[j] + beta_2*L[j]


def ST():
    x = array([-3, -1, 0, 4, 6, 8])
    y = array([0, 0, 0, 12, 18, 24])
    plot(x, y)
    
    show()
            
