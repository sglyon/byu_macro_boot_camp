# -*- coding: utf-8 -*-
# <nbformat>3</nbformat>

# <headingcell level=1>

# Spencer Lyon

# <headingcell level=4>

# Lab 24

# <headingcell level=4>

# Import Statements

# <codecell>

from __future__ import division
import scipy as sp
import numpy as np
from scipy import linalg as la
from timer import timer
import matplotlib.pyplot as plt
from numdifftools import Jacobian

# <headingcell level=4>

# Problem 1

# <codecell>

def myNewNewton(function, x0, fPrime = None, tol = 0.0001):
    if fPrime:
        xold = x0
        xnew = xold - function(xold)/fPrime(xold)
        while abs((xold- xnew)) > tol:
            xold = xnew
            xnew = xold - function(xold)/fPrime(xold)
        return xnew
    else:
        epsilon = 1e-5
        numPrime0 = (function(x0+epsilon)-function(x0))/epsilon
        xold = x0
        xnew = xold - function(xold)/numPrime0
        while abs((xold- xnew)) > tol:
            xold = xnew
            numPrimeNew = (function(xnew+epsilon)-function(xnew))/epsilon
            xnew = xold - function(xold)/numPrimeNew
        return xnew

# <codecell>

func = lambda x: x**2 -1
fPrime = lambda x: 2*x

func1 = lambda x: sp.cos(x)
func2 = lambda x: sp.sin(1/x)*x**2
func3 = lambda x: sp.sinc(x) -x

func1Prime = lambda x: -sp.sin(x)
func2Prime = lambda x: 2*x*sp.sin(1/x) - sp.cos(1/x)
func3Prime = lambda x: -sp.sin(x)/x**2 + sp.cos(x)/x -1

# <codecell>

%timeit myNewNewton(func1, 1, func1Prime)

# <codecell>

%timeit myNewNewton(func1, 1)

# <codecell>

%timeit myNewNewton(func2, 1, func2Prime)

# <codecell>

%timeit myNewNewton(func2, 1)

# <codecell>

%timeit myNewNewton(func3, 1, func3Prime)

# <codecell>

%timeit myNewNewton(func3, 1)

# <codecell>

# As you can see from the above outputs it depends on the function for 
# Whether or not it is faster with the derivative. 

# <headingcell level=4>

# Problem 2

# <codecell>

# Problem 2
funcProb2 = lambda x: x**(1/3)
# This will diverge as can easily be seen with this analtical derivation:
    # x_n+1 = x_n - (x_n**1/3)/(x_n**(-2/3)/3)
    # x_n+1 = x_n - 3*x_n
    # x_n+1 = -2*x_n // This diverges!

# <headingcell level=4>

# Problem 3

# <codecell>

funcProb3 = lambda x: x**3 - 2*x + 1/2

def findBasin(func):
    tries = sp.linspace(-2,2,200)
    ans =[]
    for i in range(0,tries.size):
        ans.append(myNewNewton(func,tries[i]))
    return [ans,tries]

# <codecell>

ans, tries = findBasin(funcProb3)
plt.figure()
plt.plot(tries,ans, 'o')
plt.show()

# <headingcell level=4>

# Problem 4

# <codecell>

def systemNewton(function, x0, fPrime = None, tol = 0.0001):
    epsilon = 1e-5
    xArray = sp.array(x0, dtype = float)
    funArray = sp.array(function)
    numEqns = funArray.size
    numVars = xArray.size
    if numVars==1:
        if fPrime:
            return myNewNewton(function, x0, fPrime)
        else:
            return myNewNewton(function, x0)
    else:
        xold = xArray
        jacfun = Jacobian(function)
        jac0 = sp.mat(jacfun.jacobian(xold))
        xnew = xold - la.solve(jac0,function(xold))
        while max(abs(xnew-xold))> tol:
            xold = xnew
            jacNew = sp.mat(jacfun.jacobian(xold))
            xnew = xold - la.solve(jacNew,function(xold))
        return xnew

# <codecell>

def matFunc(x):
    return sp.dot(sp.array([[1,2,3],[4,0,6],[7,8,9]],dtype = float),x)
matInitial = [1,2,3]
prob4Ans = systemNewton(matFunc, matInitial)
print prob4Ans

# <codecell>


