# -*- coding: utf-8 -*-
# <nbformat>3</nbformat>

# <headingcell level=1>

# Spencer Lyon

# <headingcell level=2>

# Lab 3

# <headingcell level=4>

# Import Statements

# <codecell>

import scipy as sp
from scipy import special as special

# <headingcell level=4>

# Problem 1

# <codecell>

def return42(n):
    """ Always returns the number 42"""
    return 42

def product(x,y):
    """ Return the product of two numbers """
    return x*y

def youCalled():
    """Return 'You called!'"""
    return 'You called!'

# <codecell>

return42('hey')

# <codecell>

product(3,20)

# <codecell>

youCalled()

# <headingcell level=4>

# Problem 2

# <codecell>

def sign(b):
    if b < 0:
        return -1.0
    elif b ==0 :
        return 0.0
    else:
        return 1.0

def quadrForm2(a,b,c):
    descriminant = sqrt(b**2-4*a*c)
    x1 = (-b - sign(b)*descriminant)/(2*a)
    x2 = c/(a*x1)
    return [x1, x2]
    

# <codecell>

quadrForm2(1.0,-(1.0e7 + 1.0e-7),1.0)

# <headingcell level=4>

# Problem 3

# <codecell>

def conditions(x):
    if x**2 >10 or (x>0 and x<2):
        return True
    else:
        return False
    
def sphericalBesselCondition(x):
    if abs(special.yn(1,x))>1:
        return True
    else:
        return False

# <codecell>

[conditions(-0.00001),conditions(1.999),conditions(2),conditions(4)]

# <codecell>

[[special.yn(1,0.001),special.yn(1,1)],[sphericalBesselCondition(0.001),sphericalBesselCondition(1)]]

# <headingcell level=4>

# Problem 4

# <codecell>

x = sp.linspace(1,52,52)
xmask = x
num = 0
while sp.corrcoef(x,xmask)[0,1] >.1:
    xmask = shuffle(xmask)
    num+=1
[num,xmask,sp.corrcoef(x,xmask)[0,1]]

# <codecell>


