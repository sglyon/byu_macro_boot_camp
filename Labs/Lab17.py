# -*- coding: utf-8 -*-
# <nbformat>3</nbformat>

# <headingcell level=1>

# Spencer Lyon

# <headingcell level=2>

# Lab 17

# <headingcell level=4>

# Import Statements and Workspace Setup

# <codecell>

import scipy as sp
from scipy import optimize as optim
from sympy import symbols
from scipy.special import lambertw
from scipy import integrate as spIntegrate
import matplotlib.pyplot as plt

# <codecell>

sp.set_printoptions(linewidth=140, precision=4)

# <headingcell level=4>

# Problem 1

# <codecell>

# Note, per the TA's instructions I didn't use the sp.optimize.bisect function.
def randomPoly(integ):
    coeffs = sp.random.randint(1,8,size = (1,integ)).tolist()[0]
    func = lambda x: sum(a * x**i for i, a in enumerate(coeffs))
    try:
        return optim.newton(func,.5,tol=1e-5,maxiter=500)
    except:
        print ('FAIL!')

# <codecell>

randomPoly(5)

# <headingcell level=4>

# Problem 2

# <codecell>

def myLambertW(x):
    func = lambda w: w* sp.exp(w) - x
    return optim.newton(func,5, maxiter=300)

# <codecell>

r = 11000
lambertw(r)- myLambertW(r)

# <headingcell level=4>

# Problem 3

# <codecell>

def func(y,t):
    w = y[0]
    z = y[1]
    
    f0 = z #dw/dt = ...
    f1 = -3*w #dz/dt = ...
    return [f0, f1]

# initial conditions
w0 = 1
z0 = 0
y0 = [w0, z0]
t = sp.linspace(0,5.,1000)

solution = spIntegrate.odeint(func,y0,t)
x1 = solution[:,0]
x2 = solution[:,1]
plt.figure()
plt.plot(t,x1,label='$x_1(t)$')
plt.plot(t,x2,label = '$x_2(t)$')
plt.title('My answers')
plt.legend(loc=0)

# <headingcell level=4>

# Problem 4

# <codecell>

def myNewton(f, fPrime, x0, tol=1e-3):
    xold = x0
    xnew = xold - f(xold)/fPrime(xold)
    while abs((xold- xnew)) > tol:
        xold = xnew
        xnew = xold - f(xold)/fPrime(xold)
    return xnew

# <codecell>

myNewton(lambda x: 2*x**2 - 1, lambda x:4*x, 1)

# <codecell>

optim.newton(lambda x: 2*x**2 - 1, 1,lambda x:4*x**2)

# <codecell>

optim.newton?

# <codecell>


