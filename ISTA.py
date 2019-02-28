from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import math
from pylab import *


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

def shrinkage_operator(x, i, alpha):
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
    x_i = np.copy(x_0)

    def _g(x_0, y,lbd):
        diff = 0
        for i in range(1,len(x_0)):
            diff += x[r] - [r-1]
        return np.sum((x_0-y)*(x_0-y)) + lbd * diff

    for i in range(iter):
        z_i = np.copy(x_i + beta * gradient(x_i, y, lbd))
        x_i = shrinkage_operator(z_i, i, alpha)
        print("norme(x_",i,") =", norm(x_i))
        if norm(x_i)**2 < eps:
            print("le nombre d'itérations est : ",iter)
            return True
    return False




x_0 = np.array([-3, -1, 0, 4, 6, 8])
y = np.array([0, 0, 0, 12, 18, 24])
beta = 0.01
alpha = 0.01
lbd = 0.01
iter = 1000000
eps = 0.001



print(Ista(beta, alpha, x_0, y, lbd, iter, eps))

print("x = ", x_0, "\ny= ", y)



