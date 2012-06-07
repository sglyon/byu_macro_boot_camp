# -*- coding: utf-8 -*-
# <nbformat>3</nbformat>

# <headingcell level=1>

# Spencer Lyon

# <headingcell level=2>

# Lab 1

# <headingcell level=4>

# Import Statements

# <codecell>

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as la

# <headingcell level=4>

# Problem 1

# <codecell>

x = sp.linspace(-5,5,20)
plt.plot(x,3*x,'kD')

# <headingcell level=4>

# Problem 2

# <codecell>

x2 =sp.arange(1,7)
xnew = sp.zeros((6,6))
for i in x2:
    for j in x2:
        xnew[i-1,j-1] = i*j
xnew

# <headingcell level=4>

# Problem 3

# <codecell>

vstack(sp.arange(1,7))*sp.arange(1,7)

# <headingcell level=4>

# Problem 4

# <codecell>

# Is it not just 60 x 60? If you want the total number of elements that is 60*60 = 3600
bucky = sp.loadtxt("bucky.csv", delimiter=",")
[bucky.shape, bucky.size]

# <headingcell level=4>

# Problem 5

# <codecell>

aMat = sp.random.rand(3,3)
aVec = sp.array([1,2])
aMat[:,2] = aVec

# <codecell>

aSecondMat = sp.random.rand(2,2)
sp.hstack([aMat,aSecondMat])

# <headingcell level=4>

# Problem 6

# <codecell>

h = .001
x = sp.arange(0,sp.pi,h)
approx = sp.diff(sp.sin(x**2))/h
actual = 2*sp.cos(x**2)*x
plt.figure(1)
plt.subplot(211)
plt.plot(x[:-1],approx)
plt.subplot(212)
plt.plot(x,actual)
plt.show()

# <codecell>

max(actual[:-1]-approx)

# <codecell>

plot(x[:-1],actual[:-1],x[:-1],approx,x[:-1],(actual[:-1]-approx))

# <headingcell level=4>

# Problem 7

# <codecell>

bigVec = sp.rand(1e5)
muAct = mean(bigVec)
stdAct = std(bigVec)
muEst = (1+0)/2
stdEst = (1-0)/sp.sqrt(12)
print 'Actual mean', muAct, 'Estimated mean', muEst, '\n','Actual Std', stdAct, ' Estimated Std', stdEst

# <headingcell level=4>

# Problem 8

# <codecell>

def equationMethod(A,b):
    return dot(dot(inv(dot(A.T,A)),A.T),b)

def leastSquaresMethod (A,b):
    return la.lstsq(A,b)

A = sp.randn(100,10)
b = sp.randn(100,1)


eqAns = equationMethod(A,b)
lsAns = leastSquaresMethod(A,b)
la.norm(eqAns - lsAns[0])

