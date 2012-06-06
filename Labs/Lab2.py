# -*- coding: utf-8 -*-
# <nbformat>3</nbformat>

# <headingcell level=1>

# Spencer Lyon

# <headingcell level=2>

# Lab 2

# <headingcell level=4>

# Import Statements

# <codecell>

import scipy as sp
import numpy as np
from scipy import linalg as la
import matplotlib.pyplot as plt
from scipy import sparse as spar
from scipy.sparse import linalg as sparla

# <headingcell level=4>

# Problem 1

# <codecell>

def flatDiags(n):
    x = sp.zeros((n,n))
    for i in range(n):
        x +=  sp.diagflat(sp.ones(n-i)*(i+1),i)
    return x
    

# <codecell>

flatDiags(5)

# <codecell>

def totalDiags(n):
    x = sp.zeros((n,n))
    for i in range(n):
        x+= sp.diagflat(sp.ones(n-i)/float((i+1)),i) + sp.diagflat(sp.ones(n-i)/float((i+1)),-i)
    x -= sp.diag(sp.ones(n),0)
    return x

# <codecell>

totalDiags(5)

# <headingcell level=4>

# Problem 2

# <codecell>

def toepOne(n):
    return la.triu(la.toeplitz(sp.arange(1,n+1,1)))

# <codecell>

toepOne(5)

# <codecell>

def toepTwo(n):
    return la.toeplitz(1/sp.arange(1,n+1,1))

# <codecell>

toepTwo(5)

# <codecell>

def diagFlatThree(n):
    return sp.diagflat(sp.arange(1,n+1,1))

# <codecell>

diagFlatThree(5)

# <headingcell level=4>

# Problem 3

# <codecell>

def randomMatrix(n):
    x = diagflat(sp.ones(n),0)
    x[:,-1]    = randn(n)
    x[:,-2]    = randn(n)
    x[-1,:]    = randn(n)
    x[-2,:]    = randn(n)
    x[n-2,n-2] = 1
    x[n-1,n-1] =1
    return x

# <codecell>

randomMatrix(5)

# <headingcell level=4>

# Problem 4

# <codecell>

def triDiagonal(n):
    return la.toeplitz(sp.hstack([sp.array([2,-1]), sp.zeros(n-2)]))

# <codecell>

triDiagonal(5)

# <headingcell level=4>

# Problem 5

# <codecell>

def sparse2ndDev(n):
    x = sp.vstack([sp.ones(n)*2, sp.ones(n)*-1, sp.ones(n)*-1])
    return spar.spdiags(x, [0,1,-1], n, n)

# <codecell>

sparse2ndDev(10)

# <codecell>

sparse2ndDev(10).todense()

# <headingcell level=4>

# Problem 6

# <codecell>

def sparSolve(n):
    A = sparse2ndDev(n)
    b = sp.rand(n,1)
    return sparla.spsolve(A,b)

# <codecell>

def denseSolve(n):
    A = triDiagonal(n)
    b = sp.rand(n,1)
    return la.solve(A,b)

# <headingcell level=6>

# Sparse Results

# <codecell>

%timeit sparSolve(10)

# <codecell>

%timeit sparSolve(20)

# <codecell>

%timeit sparSolve(50)

# <codecell>

%timeit sparSolve(100)

# <codecell>

%timeit sparSolve(500)

# <codecell>

%timeit sparSolve(1200)

# <headingcell level=6>

# Dense Results

# <codecell>

%timeit denseSolve(10)

# <codecell>

%timeit denseSolve(20)

# <codecell>

%timeit denseSolve(50)

# <codecell>

%timeit denseSolve(100)

# <codecell>

%timeit denseSolve(500)

# <codecell>

%timeit denseSolve(1200)

# <headingcell level=4>

# Problem 7

# <codecell>

vals, vecs = la.eig(sparse2ndDev(1200).todense())
a = vals.min()
a*1200**2

# <codecell>

#This approaches pi^2 as n gets larger

# <codecell>

sp.pi**2

