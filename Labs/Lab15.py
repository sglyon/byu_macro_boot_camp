# -*- coding: utf-8 -*-
# <nbformat>3</nbformat>

# <headingcell level=1>

# Spencer Lyon

# <headingcell level=2>

# Lab 15

# <headingcell level=4>

# Import Statements

# <codecell>

import scipy as sp
from scipy import linalg as la
import RowOperations as rop
from timer import timer
# Importing the LUDecomp function was being funny so I just copied and pasted it below

# <codecell>

def LUDecomp(mat):
    n = mat.shape[0]
    EL = []
    L = sp.eye(n)
    U = mat
    ops = 0
    # Construct all type 3 matricies
    for col in range(0,n):
        for row in range(col+1,n):
            E = rop.cmultadd(n,row,col,(-U[row,col]/U[col,col]))
            ops+=1
            
            E1= rop.cmultadd(n,row,col, U[row,col]/U[col,col])
            ops+=1
            
            U =sp.dot(E,U)
            ops+= 2*size(U)**2
            
            EL.append(E1)
            
    # Construct all type 1 matrcies.
    for j in range(0,n):
        E = rop.cmult(n,j,1/U[j,j])
        ops+=1
        
        E1 = rop.cmult(n,j,U[j,j])
        ops += 1
        
        U = sp.dot(E,U)
        ops+= 2*size(U)**2
        
        EL.append(E1)
        
    for i in EL:
        L = sp.dot(L,i)
        ops+= 2*size(L)**2
        
    return [L,U,ops]

# <headingcell level=4>

# Problem 1

# <codecell>

# As L will be lower triangular it makes sense to start with a matrix of zeros and populate the entires below the diagonal.
# First element of L (L[1,1]) = sp.sqrt(A[1,1])
# We would then move accross row 1 until we hit the diagonal. In this case we are already at it so we move to row 2
# For row two we calculate each entry using L[2,j] = 1/L[j,j]*(A[1,2] - sum(L[i,k]*L[j,k],{k,1,i-1})
# We would do this until we hit the diagonal (until j=i) and then move to the next row and repeat

# <headingcell level=4>

# Problem 2

# <codecell>

def CholeskyDecomp(A):
    """ 
    Run a Cholesky Decomposition on the matrix A
    """
    m = shape(A)[0]
    L = sp.zeros((m,m))
    ops = 0
    
    # Iterate one row at a time
    for row in range(0, m):
        
        
        # Go through the current row and fill in each column one at a time
        # Notice we stop at 'row' because that is where we hit the diagonal.
        for col in range(0, row):
            const = 1 / L[col, col]
            ops+=1
            
            aMatVal = A[row, col]
            theSum = 0
            
            # Now populate theSum according to the first formula
            for k in range(0, col):
                theSum += L[row, k] * L[col, k]
                ops += 2
            
            L[row, col] = const * (aMatVal - theSum)
            ops += 2
            
            # Now we need to make sure we get the diagonal entries right using the second formula
            
        aDiagVal = A[row, row]
        theDiagSum = 0
        for k in range(0, row):
            theDiagSum += L[row, k] ** 2 #* L.T[row, k]
            ops += 2
        
        L[row, row] = (aDiagVal - theDiagSum) ** 0.5
        ops += 2
    
    return [L, ops]

# <codecell>

b = sp.rand(3,3)
B = sp.dot(b.T,b)
CholeskyDecomp(B)[0]

# <headingcell level=4>

# Problem 3

# <codecell>

c = sp.rand(10,10)
C = sp.dot(c.T,c)

# <codecell>

% timeit CholeskyDecomp(C)

# <codecell>

%timeit LUDecomp(C)

# <codecell>

def CholeskyDecompA(C):
    return CholeskyDecomp(C)

def LUArray(A):
    return la.lu(A)

# <codecell>

cholOps100 = CholeskyDecomp(C)[1]
LUops100 = LUDecomp(c)[2]

# <codecell>

LUops10 = LUDecomp(c)[2]
cholOps10 = CholeskyDecomp(C)[1]

# <codecell>

LUops100/cholOps100

# <codecell>

LUops10/cholOps10

# <headingcell level=5>

# As can be seen for a 100 x 100 matrix it takes the LU algorithm 5.7M times more operations than the Cholesky algorithm. For a 10 x 10 matrix the LU algorithm takes 4.5k times the number of operations

