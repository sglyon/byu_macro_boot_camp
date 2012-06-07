# -*- coding: utf-8 -*-
# <nbformat>3</nbformat>

# <headingcell level=1>

# Spencer Lyon

# <headingcell level=3>

# Lab 13

# <headingcell level=4>

# Import Statements

# <codecell>

import scipy as sp
from scipy import linalg as la
from Lab9 import qrDecomp

# <headingcell level=4>

# Problem 1

# <codecell>

# Two matricies A and B are similar if they can be written as B = P^-1 A P
# If we let P = Q (which is always invertible because the columns of Q form an orthonormal set)
# B = Q^-1 (Q R) * Q = I R Q = R Q

# <headingcell level=4>

# Problem 2

# <codecell>

a = sp.randn(4,4)
A = sp.dot(a.T,a);A

# <codecell>

def qrEigenSolve(mat,iterations):
    """ 
    Use the QR algorithms to solve for all the eigenvalues of the input matrix mat.

    The return is a vector of eigenvalues
    """
    n = iterations;
    A = mat
    runs = 0
    QList = []
    U = sp.eye(shape(A)[0])
    
    while runs < n:
        Q,R = la.qr(A, mode = 'economic')
        A = sp.dot(R,Q)
        U = sp.dot(U,Q)
        runs+=1
        
    
    # for row in range(0,shape(U)[0]):
    #     U[row,:] = U[row,:]/la.norm(U[row,:])
        
    return sp.diag(A),U

# <codecell>

vals, vecs = qrEigenSolve(A,10000)
[vals, vecs]

# <codecell>

la.eig(A)

# <headingcell level=4>

# Problem 3

# <codecell>

B = sp.rand(5,5);B

# <codecell>

sp.sort(qrEigenSolve(B,100000)[0])

# <codecell>

sp.sort(la.eig(B)[0])

# <headingcell level=6>

# As can be seen above my algorithm does not find complex eigenvalues

# <headingcell level=6>

# The real eigenvalues are the same

# <headingcell level=4>

# Problem 4

# <headingcell level=6>

# For this problem I just added the line 'U = sp.dot(U,Q)' inside the while loop.
# Also we needed the columns of U as the eigenvectors so to have that return I asked for U.T

# <codecell>


