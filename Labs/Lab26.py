# Spencer Lyon
# Lab 26
from __future__ import division
import scipy as sp
from scipy import linalg as la
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------#
# Problem 1
def secant(func, x1, x2, tol=1e-5):
    """
    Use the secant method to find the root of a given funciton
    
    Inputs:
        func: the function for which you want the zero
        x1: The lower bound in which to search
        x2: The upper bound in which to search
        tol: The tolerance level for approximation
    
    Output:
        x: The argument that forces func to zero
        xlist: A running list of all x's the algorithm went through. 
        
    Exceptions:
        If there is no zero between x1,x2 an error will be raised.
    """
    if abs(func(x1)) < tol:
        return x1
    if abs(func(x2)) < tol:
        return x2
    
    xnew  = 10
    i = 0
    xlist = []
    while abs(func(xnew))> tol and i<200:
        xnew = x1 - (func(x1)*(x1-x2))/(func(x1)-func(x2))
        x1 = x2
        x2 = xnew
        i += 1
        xlist.append(xnew)
    if i == 200:
        raise ValueError('No using initial guesses. Try something else')
    else:
        return xnew, xlist
    
f = lambda x: sp.exp(x) - 2

theWinningX, theList = secant(f, 1, 5)

els = len(theList)

xs = sp.log(theList[1:els - 2]) - theWinningX
ys = sp.log(theList[2:els - 1]) - theWinningX

slope = 10* (ys[1] - ys[0])/(xs[1] - xs[0]); print ' Slope =   ', slope
plt.plot(xs,ys)
plt.show()

#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
# Problem 2


def regulaFalsi(func, x1, x2, tol = 1e-12):
    if abs(func(x1)) <tol:
        return x1
    if abs(func(x2)) < tol:
        return x2
    xnew = 3
    
    if  func(x1)*func(x2) >=0:
        print 'you have chosen bad starting points. They have the same sign'
    else:
        while abs(func(xnew)) > tol:
            xnew = x1 - (func(x1)*(x1-x2))/(func(x1)-func(x2))
            if func(xnew) * func(x2) <0:
                x1 = xnew
                x2 = x2
            else:
                x1 = x1
                x2 = xnew
        return xnew

f2 = lambda x: sp.sin(x)

print 'RF answer:',  regulaFalsi(f2, 2.0, 6.0), 'func eval at ans',\
f2(regulaFalsi(f2, 2.0, 6.0))

ftime1 = lambda x: sp.exp(x) - 1 # use -5, 2
ftime2 = lambda x: sp.cos(x) # use 2, 6
ftime3 = lambda x: x**7 # use -2, 2

# Do timing stuff in notebook



#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
# Problem 3

def broyden (func, x1, x2, tol = 1e-8):
    """
    Implementation of the Broyden method finding roots to a multivariable func-
    tion.
    
    Inputs:
        func: The vector function you wish to find a root for
        x1: One initial guess vector
        x2: A second initial guess vector
        [tol]: optional. Tolerance level
        
    Returns:
        x such that func(x) = 0
    """
    if abs(func(x1)) < tol:
        return x1
    if abs(func(x2)) < tol:
        return x2
    
    x1Array = sp.array(x1)
    x2Array = sp.array(x2)
    ndims = x1Array.size
    Jacob = sp.eye(ndims) + sp.rand(ndims, ndims)
    
    xnew = sp.rand(ndims)
    
    i=0
    while abs(la.norm(func(xnew))) > tol:
        f1 = func(x1)
        f2 = func(x2)
        
        deltax = (x2-x1)
        
        Jnew = Jacob + (f2- f1 - sp.dot(sp.dot(Jacob,deltax),deltax.T))\
        /(la.norm(deltax)**2)
        
        xnew = la.inv(Jnew) * (-f2) + x2
        
        x1 = x2
        x2 = xnew
        Jacob = Jnew
        
        i += 1
        print i
    return xnew
        

func3 = lambda x: sp.cos(x[0] * x[1]) + x[0] * x[2]**3 + x[1]**2 \
- x[1] + x[0]**2 + x[2] +x[0]

x1 = sp.rand(3)
x2 = sp.rand(3)
print broyden(func3, x1, x2)
#------------------------------------------------------------------------------#
# Problem 4

def modifiedBroyden (func, x1, x2, tol = 1e-8):
    """
    Implementation of the Broyden method finding roots to a multivariable func-
    tion.
    
    Inputs:
        func: The vector function you wish to find a root for
        x1: One initial guess vector
        x2: A second initial guess vector
        [tol]: optional. Tolerance level
        
    Returns:
        x such that func(x) = 0
    """
    if abs(func(x1)) < tol:
        return x1
    if abs(func(x2)) < tol:
        return x2
    
    x1Array = sp.array(x1)
    x2Array = sp.array(x2)
    ndims = x1Array.size
    JoldInv = la.inv(sp.eye(ndims) + sp.rand(ndims, ndims))
    
    xnew = sp.rand(ndims)
    
    i=0
    while abs(la.norm(func(xnew))) > tol:
        f1 = func(x1)
        f2 = func(x2)
        deltaf = (f2 - f1)
        
        
        deltax = (x2-x1)
        
        JnewInv = JoldInv + (deltax - sp.dot(JoldInv, deltaf))\
        /(sp.dot(deltax.T, JoldInv))
        
        xnew = JnewInv * (-f2) + x2
        
        x1 = x2
        x2 = xnew
        JoldInv = JnewInv
        
        i += 1
        print i
    return xnew
        

func4 = lambda x: sp.cos(x[0] * x[1]) + x[0] * x[2]**3 + x[1]**2 \
- x[1] + x[0]**2 + x[2] +x[0]

modifiedBroyden(func4, x1,x2)