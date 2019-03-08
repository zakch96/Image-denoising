from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import math
from pylab import *
import sys 
from termcolor import colored, cprint 


def norm(x):
    nrm =0
    for i in range(len(x)):
        nrm += pow(x[i], 2)
    return sqrt(nrm)

def partial_derivative(func, var=0, point=[]):
    args = point[:]
    def wraps(x):
        args[var] = x
        return func(*args)
    return misc.derivative(wraps, point[var], dx = 1e-6)


def sgn(a):
    signe = 0
    if  a > 0:
        signe = 1
    elif a < 0:
        signe = -1
    else:
        return 0
    return signe

def shrinkage_operator(x, alpha):
    for i in range(len(x)):
        x[i] = abs((abs(x[i]) - alpha)) * sgn(x[i])
    return x

def gradient(x, y, lbd):
    grad = np.ones(len(x))
    for r in range(len(x)):
        grad[r] = 2*(x[r] - y[r]) + lbd*sgn(x[r])
    return grad

def Ista(beta, alpha, x_0, y, lbd, iter, eps):
    n = len(x_0)
    x_i = ones(n) 
    z_i = ones(n)
    x_i = ones(n)

    def _g(x_0, y,lbd):
        diff = 0
        for r in range(1,len(x_0)):
            diff += x_0[r] - x_0[r-1]
        return np.sum((x_0-y)*(x_0-y)) + lbd * diff

    for i in range(iter):
        z_i = np.copy(x_i + beta * gradient(x_i, y, lbd))
        x_i = shrinkage_operator(z_i, alpha)
        print("norme(x_",i,") =", norm(x_i))
        print("g(x_",i,") =", _g(x_i, y, lbd))
        if norm(x_i-x_0)**2 < eps:
            print("le nombre d'itérations est : ", i)
            return True
    if i == iter-1:
        cprint("Erreur, le nombre d'itérations est dépassé sans convergence", 'red', attrs=['bold'], file=sys.stderr)
    return False

# def Fista(beta, alpha, x_0, y, lbd, iter, eps):
#     n = len(x_0)
#     x_i = ones(n)
#     x_copy = ones(n)
#     z_i = np.copy(x_0)
#     x_i = np.copy(x_0)

#     for i in range(iter):
#         x_i = 






x_0 = np.random.rand(10)  
y = np.random.rand(10) 
beta = -0.0001
alpha = 0.001
lbd = 0.01
iter = 500000
eps = 1

print("y before calling ISTA = ", y)

print(Ista(beta, alpha, x_0, y, lbd, iter, eps))

print("x = ", x_0, "\ny = ", y, "\nerreur = ", sqrt(norm((y-x_0)**2)))



