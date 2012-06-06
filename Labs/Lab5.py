# -*- coding: utf-8 -*-
# <nbformat>3</nbformat>

# <headingcell level=1>

# Spencer Lyon

# <headingcell level=2>

# Lab 5

# <headingcell level=4>

# Import Statements

# <codecell>

import scipy as sp
from __future__ import division

# <headingcell level=4>

# Problem 1

# <codecell>

def sumOdd(n):
    """
    This function sums all the odd numbers less than n
    """
    runSum=0
    for i in range(1,n+1):
        if i%2!=0:
            runSum+=i
    return runSum

# <codecell>

[sumOdd(5),1+3+5,sumOdd(11),1+3+5+7+9+11]

# <headingcell level=4>

# Problem 2

# <codecell>

def expSS(x,sumLimit):
    """
    Uses a Taylor series approximaiton to determine exp(x)
    Inputs:
        x = the number to be evaluated
        sumLimit = how far out the inifite sum should go
    Outputs:
        ans = the estimated value for exp(x)
    """

    divides = 0
    ans = 0
    
    while x>.001:
        x/=2
        divides+=1

    for i in range(sumLimit):
        ans+=(x**i)/sp.math.factorial(i)
    for i in range(0,divides):
        ans = ans**2
    return ans

# <codecell>

[['My Approximation', 'Built in function','  Difference  '],[expSS(3,20),exp(3),expSS(3,20)-sp.exp(3)],[expSS(30,50),sp.exp(30),expSS(30,80)-sp.exp(30)]]

# <headingcell level=4>

# Problem 3

# <codecell>

def expSSVEctorInput(W,sumLimit):
    """
    Uses a Taylor series approximaiton to determine exp(x) for each x in W
    Inputs:
        W = A 1-dimensional numpy array of numbers to be evaluated
        sumLimit = how far out the inifite sum should go
    Outputs:
        ans = the estimated value for exp(x)

    Example:
    >>> expSSVectorInput(sp.array([3,4]),30)
    array([ 20.08553692,  54.59815003])
    """

    divides = 0
    ans = 0
    
    while max(W)>.001:
        W=W/2
        divides+=1

    for i in range(sumLimit):
        ans+=(W**i)/sp.math.factorial(i)
    for i in range(0,divides):
        ans = ans**2
    return ans

# <codecell>

expSSVEctorInput(sp.array([3,4]),30)

