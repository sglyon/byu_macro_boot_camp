# -*- coding: utf-8 -*-
# <nbformat>3</nbformat>

# <headingcell level=1>

# Spencer Lyon

# <headingcell level=2>

# Lab 4

# <headingcell level=4>

# Import Statements

# <codecell>

import RowOperations as rop
import scipy as sp
from __future__ import division
from scipy import linalg as la

# <headingcell level=4>

# Problem 1

# <codecell>

A = sp.array([[4,5,6,3],[2,4,6,4],[7,8,0,5]])
A1 = sp.dot(rop.cmultadd(3,1,0, - A[1,0]/ A[0,0]), A);
A2 = sp.dot(rop.cmultadd(3,2,0, -A1[2,0]/A1[0,0]), A1)
A3 = sp.dot(rop.cmultadd(3,2,1, -A2[2,1]/A2[1,1]), A2)

# <codecell>

def rowReduce(matrix):
    n = matrix.shape[0]
    for i in range(0,n):
        for j in range(i,n):
            matrix = sp.dot(rop.cmultadd(n,j,i,-matrix[j,i]/matrix[i,i]),matrix)
    for i in range(0,n):
        matrix[i,:]/=matrix[i,i]
    return matrix

# <codecell>

rowReduce(A)

# <headingcell level=4>

# Problem 2

# <codecell>

def LUDecomp(mat):
    n = mat.shape[0]
    EL = []
    L = sp.eye(n)
    U = mat
    # Construct all type 3 matricies
    for col in range(0,n):
        for row in range(col+1,n):
            E = rop.cmultadd(n,row,col,(-U[row,col]/U[col,col]))
            E1= rop.cmultadd(n,row,col, U[row,col]/U[col,col])
            U =sp.dot(E,U)
            EL.append(E1)
            
    # Construct all type 1 matrcies.
    for j in range(0,n):
        E = rop.cmult(n,j,1/U[j,j])
        E1 = rop.cmult(n,j,U[j,j])
        U = sp.dot(E,U)
        EL.append(E1)
        
    for i in EL:
        L = sp.dot(L,i)
        
    return [L,U]

# <codecell>

B = sp.rand(3,3)
LB, UB = LUDecomp(B)
[LB,UB, sp.dot(LB,UB)-B]

# <headingcell level=4>

# Problem 3

# <codecell>

def detLU(mat):
    L,U = LUDecomp(mat)
    sumL = 1
    sumU = 1
    for i in range(0,shape(L)[0]):
        sumL *= L[i,i]
        sumU *= U[i,i]
    return sumL*sumU

# <codecell>

detLU(B)

# <codecell>

la.det(B)

