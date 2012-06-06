# -*- coding: utf-8 -*-
# <nbformat>3</nbformat>

# <headingcell level=1>

# Spencer Lyon

# <headingcell level=3>

# Lab 9

# <headingcell level=4>

# Import Statements

# <codecell>

import scipy as sp
from scipy import linalg as la
from numpy import shape

# <headingcell level=4>

# Problem 1

# <codecell>

b = sp.random.randint(-5,5,(7,5))

# <codecell>

def qrDecomp(mat):
    n = mat.shape[0]
    numQs = shape(mat)[1]
    qList = []
    xList = []
    
    for x in range(0,numQs):
        xList.append(mat[:,x])
    
    q1 = (xList[0]/la.norm(xList[0]))*-1
    qList.append(q1)
    
    for qi in range(1,numQs):
        for xi in range(qi,numQs):
            xList[xi] = xList[xi] - sp.dot(sp.dot(xList[xi],qList[qi-1]),qList[qi-1])
        qList.append(xList[qi]/la.norm(xList[qi]))
        
    Q = sp.zeros((n,numQs))
    for j in range(0,shape(qList)[0]):
        Q[:,j] = qList[j]
        
    R = sp.dot(Q.T,mat)
    return [Q, R]
                     
 

# <codecell>

Qb,Rb = qrDecomp(b)
[Qb, Rb]

# <codecell>

la.qr(b, mode = 'economic')

# <codecell>

# My output and the output from the built in function are the same except for some roundoff error with my entries of R not showing as zero.

# <codecell>


