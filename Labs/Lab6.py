# -*- coding: utf-8 -*-
# <nbformat>3</nbformat>

# <headingcell level=1>

# Spencer Lyon

# <headingcell level=2>

# Lab 6

# <headingcell level=4>

# Import Statements

# <codecell>

import scipy as sp
from timer import timer
from matplotlib import pyplot as plt
import scipy.linalg as la

# <headingcell level=4>

# Problem 1

# <codecell>

def addArray(A,B):
    return A+B

def invArray(A):
    return la.inv(A)

def detArray(A):
    return la.det(A)

def SVDArray(A):
    return la.svd(A)

def LUArray(A):
    return la.lu(A)

def solveSystem(A,b):
    return la.solve(A,b)

# <codecell>

i = sp.arange(1500,2500+200,200)
k = 1
y = []
ans = 0
for n in i:
    with timer(loops =20) as t:
        ans = t.time(addArray, sp.rand(n,n), sp.rand(n,n))
        y.append(ans[0][0])
X =  sp.row_stack([sp.log(i), sp.ones_like(i)]).T
sol = la.lstsq(X, sp.log(y))
print sol[0][0]
plt.loglog(i,y)
plt.show()

# <codecell>

i = sp.arange(1500,2500+200,200)
k = 1
y = []
ans = 0
for n in i:
    with timer(loops =20) as t:
        ans = t.time(invArray, sp.rand(n,n))
        y.append(ans[0][0])
X =  sp.row_stack([sp.log(i), sp.ones_like(i)]).T
sol = la.lstsq(X, sp.log(y))
print sol[0][0]
plt.loglog(i,y)
plt.show()

# <codecell>

i = sp.arange(1500,2500+200,200)
k = 1
y = []
ans = 0
for n in i:
    with timer(loops =10) as t:
        ans = t.time(detArray, sp.rand(n,n))
        y.append(ans[0][0])
X =  sp.row_stack([sp.log(i), sp.ones_like(i)]).T
sol = la.lstsq(X, sp.log(y))
print sol[0][0]
plt.loglog(i,y)
plt.show()

# <codecell>

i = sp.arange(1500,2500+200,200)
k = 1
y = []
ans = 0
for n in i:
    with timer(loops =10) as t:
        ans = t.time(SVDArray, sp.rand(n,n))
        y.append(ans[0][0])
X =  sp.row_stack([sp.log(i), sp.ones_like(i)]).T
sol = la.lstsq(X, sp.log(y))
print sol[0][0]
plt.loglog(i,y)
plt.show()

# <codecell>

i = sp.arange(1500,2500+200,200)
k = 1
y = []
ans = 0
for n in i:
    with timer(loops =10) as t:
        ans = t.time(LUArray, sp.rand(n,n))
        y.append(ans[0][0])
X =  sp.row_stack([sp.log(i), sp.ones_like(i)]).T
sol = la.lstsq(X, sp.log(y))
print sol[0][0]
plt.loglog(i,y)
plt.show()

# <codecell>

i = sp.arange(1500,2500+200,200)
k = 1
y = []
ans = 0
for n in i:
    with timer(loops =10) as t:
        ans = t.time(solveSystem, sp.rand(n,n), sp.rand(n,1))
        y.append(ans[0][0])
X =  sp.row_stack([sp.log(i), sp.ones_like(i)]).T
sol = la.lstsq(X, sp.log(y))
print sol[0][0]
plt.loglog(i,y)
plt.show()

# <headingcell level=4>

# Problem 2

# <codecell>

n = 100
b = sp.rand(n,1)
u = sp.rand(n,1)
v = sp.rand(1,n)
A = sp.eye(n) + sp.dot(u,v)


def sherMorWoodSol(A,b,n,u,v):
    Ainv = sp.eye(n) - sp.dot(sp.dot(sp.eye(n),u),v)/(1+sp.dot(v,u))
    return sp.dot(Ainv,b)

def invSys(A,b):
    return sp.dot(la.inv(A),b)

with timer(loops=10) as t:
    t.time(solveSystem, A,b)
    t.time(invSys, A,b)
    t.time(sherMorWoodSol, A,b,n,u,v)
    t.printTimes()

# <codecell>


