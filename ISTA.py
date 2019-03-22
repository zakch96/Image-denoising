import numpy as np
import matplotlib.pyplot as plt
import math
from pylab import *
import time 
import matplotlib.pyplot as plt




def norm_1(x):
	return np.sum.abs(x)

def norm_2(x):
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
    #l = np.copy(x)
    for i in range(len(x)):
        x[i] = abs((abs(x[i]) - alpha)) * sgn(x[i])
    return x

def gradient(x, y, lbd):
    grad = np.ones(len(x))
    for r in range(len(x)):
        grad[r] = 2*(x[r] - y[r]) + lbd*sgn(x[r])
    return grad

def Ista(beta, alpha, x_0, y, lbd, iter, eps, norm):
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
            return x_i
        if i == iter-1:
        	print("le nombre d'itérations est : ", i)
        	print("après ISTA x_i =",x_i,"\nl'erreur = ", sqrt((norm(x_i-x_0)))/n)
        	#cprint("Erreur, le nombre d'itérations est dépassé sans convergence", 'white', attrs=['bold'], file=sys.stderr)
        #print("ITERATION ", i, 'norme(x_', i,'-x_0)**2 = ', norm(x_i-x_0)**2)

    print("le nombre d'itérations est i = ", i)
    print("norme(x_",i,"-x_0)^2 =", norm(x_i-x_0))
    x_0 = x_i
    return x_i

def Fista(beta, alpha, x_0, b, lbd, lf, iter, eps, norm):
	k = 0
	n = len(x_0)
	x_i = ones(n)
	x_i_prev = ones(n)/2.
	y_i = ones(n)/2.
	t_i = 1.
	t_i_prev = 1.

	for i in range(iter):
		#x_i = shrinkage_operator(z_i, alpha)
		y_i = (1./(lf+lbd**2)) * (lf * y_i -(y_i - b))
		x_i = shrinkage_operator(b, alpha)
		t_i = (1 + sqrt(1 + 4 * t_i_prev**2 )) / 2.
		tmp = (t_i_prev - 1)/t_i
		y_i = x_i + tmp * (x_i - x_i_prev)
		if norm(x_i-x_0)**2 < eps:
			print("après ISTA x_i =",x_i,"\nl'erreur = ", norm(x_i-x_0)**2)
			return x_i
		x_i_prev = np.copy(x_i)
		t_i_prev = t_i
		if i == iter-1:
			print("le nombre d'itérations est : ", i)
			print("après FISTA x_i =",x_i,"\nl'erreur = ", norm(x_i-x_0)**2)
		    #cprint("Erreur, le nombre d'itérations est dépassé sans convergence", 'red', attrs=['bold'], file=sys.stderr)
		#print("ITERATION ", i, 'norme(x_', i,'-x_0)**2 = ', norm(x_i-x_0)**2)
		k=i

	print("le nombre d'itérations est : ", k)
	print("norme(x_",i,"-x_0)^2 =", norm(x_i-x_0))
	return x_i

def soft_thresh(x, l):
	return np.sign(x) * np.maximum(np.abs(x) - l, 0.)

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
        x = soft_thresh(z, l / L)
        t0 = t
        t = (1. + sqrt(1. + 4. * t ** 2)) / 2.
        z = x + ((t0 - 1.) / t) * (x - xold)
        this_pobj = 0.5 * linalg.norm(A.dot(x) - b) ** 2 + l * linalg.norm(x, 1)
        pobj.append((time.time() - time0, this_pobj))

    times, pobj = map(np.array, zip(*pobj))
    return x




A = np.eye(10)
# for i in range(9):
# 	A[i][i+1]= 1
x_0 = zeros(10) 
b = np.random.normal(loc=0.0, scale=0.5, size=10)
#b = 
beta = -0.000001
alpha = 0.0001
lbd = 1
lf = 1.5
iter1 = 30000
iter2 = 30000
eps = 0.0000001


print('==========Ista RESULTS==========')
print("__________'NORM_1'__________")
print("b before calling ISTA = ", b)
print(Ista(beta, alpha, x_0, b, lbd, iter1, eps, norm_2))
print("\n")

print("__________'NORM_2'__________")
print("b before calling ISTA = ", b)
print(Ista(beta, alpha, x_0, b, lbd, iter1, eps, norm_2))
print("\n")

print('==========Fista RESULTS==========')
print("__________'NORM_1'__________")
print(Fista(beta, alpha, x_0, b, lbd, lf, iter2, eps, norm_2))
print("\n")

print("___________NORM_2'__________")
print(Fista(beta, alpha, x_0, b, lbd, lf, iter2, eps, norm_2))
print("\n")

print("__________'FISTA RESULTS BIS'__________")
print(fista(A, b, 1, 10))
print("\n")

fista_bis = fista(A, b, 1, 10)

from pylab import *

plot(b)
plot(fista_bis)
show()
