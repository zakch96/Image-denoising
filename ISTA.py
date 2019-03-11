from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import math
from pylab import *
from scipy import signal
from termcolor import colored, cprint
import time 


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
    l = np.copy(x)
    for i in range(len(x)):
        l[i] = abs((abs(x[i]) - alpha)) * sgn(x[i])
    return x

def gradient(x, y, lbd):
    grad = np.ones(len(x))
    for r in range(len(x)):
        grad[r] = 2*(x[r] - y[r]) + lbd*sgn(x[r])
    return grad

def Ista(beta, alpha, x_0, y, lbd, iter, eps):
    n = len(x_0)
    z_i = zeros(n)
    x_i = zeros(n)

    print("Avant ISTA y = ", y)
    def _g(x_0, y,lbd):
        diff = 0
        for r in range(1,len(x_0)):
            diff += x_0[r] - x_0[r-1]
        return np.sum((x_0-y)*(x_0-y)) + lbd * diff

    for i in range(iter):
        z_i = np.copy(x_i + beta * gradient(x_i, y, lbd))
        x_i = shrinkage_operator(z_i, alpha)
        #print("g(x_",i,") =", _g(x_i, y, lbd))
        if _g(x_0, y,lbd)**2 < eps:
            print("le nombre d'itérations est : ", i)
            print("après ISTA x_i =",x_i,"\nl'erreur = ", sqrt((norm(x_i-x_0)))/n)
            return True
    if i == iter-1:
    	print("le nombre d'itérations est : ", i)
    	print("après ISTA x_i =",x_i,"\nl'erreur = ", sqrt((norm(x_i-x_0)))/n)
    	cprint("Erreur, le nombre d'itérations est dépassé sans convergence", 'white', attrs=['bold'], file=sys.stderr)
    print("norme(x_",i,"-x_0)^2 =", norm(x_i-x_0))
    return False

def Fista(beta, alpha, x_0, b, lbd, lf, iter, eps):
	i = 0
	n = len(x_0)
	x_i = zeros(n)
	x_i_prev = zeros(n)
	y_i = zeros(n)
	t_i = 1.
	t_i_prev = 1.
	for i in range(iter):
		#x_i = shrinkage_operator(z_i, alpha)
		x_i = np.copy((1./(lf+lbd**2)) * (lf * y_i -(y_i - b)))
		t_i = (1 + sqrt(1 + 4 * t_i_prev**2 )) / 2.
		tmp = (t_i_prev - 1)/t_i
		y_i = x_i + tmp * (x_i - x_i_prev)
		if norm(x_i-x_0)**2 < eps:
			print("après ISTA x_i =",x_i,"\nl'erreur = ", norm(x_i-x_0)**2)
			print("le nombre d'itérations est : ", i)
			return True
		x_i_prev = np.copy(x_i)
		t_i_prev = t_i
	if i == iter-1:
		print("après FISTA x_i =",x_i,"\nl'erreur = ", norm(x_i-x_0)**2)
		cprint("Erreur, le nombre d'itérations est dépassé sans convergence", 'red', attrs=['bold'], file=sys.stderr)
	return False

def fista(A, b, l, maxit):
    x = np.zeros(A.shape[1])
    pobj = []
    t = 1
    z = x.copy()
    L = linalg.norm(A) ** 2
    time0 = time.time()
    for i in range(maxit):
        xold = x.copy()
        z = z + A.T.dot(b - A.dot(z)) / L
        x = shrinkage_operator(z, l / L)
        t0 = t
        t = (1. + sqrt(1. + 4. * t ** 2)) / 2.
        z = x + ((t0 - 1.) / t) * (x - xold)
        this_pobj = 0.5 * linalg.norm(A.dot(x) - b) ** 2 + l * linalg.norm(x, 1)
        pobj.append((time.time() - time0, this_pobj))

    times, pobj = map(np.array, zip(*pobj))
    return x




A = np.eye(10)
x_0 = zeros(10) 
y = np.copy(x_0 + np.random.normal(loc=0.0, scale=0.2, size=10) )
y[5] = 1.
beta = -0.00001
alpha = 0.1
lbd = 0.1
lf = 1.5
iter = 10000
eps = 0.001

print("y before calling ISTA = ", y)
print(Ista(beta, alpha, x_0, y, lbd, iter, eps))
print(Fista(beta, alpha, x_0, y, lbd, lf, iter, eps))
print(fista(A, y, 2, 100))

