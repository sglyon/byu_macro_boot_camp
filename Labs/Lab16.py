# -*- coding: utf-8 -*-
# <nbformat>3</nbformat>

# <headingcell level=1>

# Spencer Lyon 

# <headingcell level=2>

# Lab 16

# <headingcell level=4>

# Import Statements

# <codecell>

import scipy as sp
import scipy.linalg as la
import matplotlib.pyplot as plt
import numpy.linalg as nla

# <headingcell level=4>

# Problem 1

# <codecell>

X = sp.misc.imread('/Users/spencerlyon2/Desktop/fingerprint.png')[:,:,0].astype(float)
startRank = nla.matrix_rank(X); startRank
[X.nbytes, startRank]

# <codecell>

U,s,Vt = la.svd(X)
S = sp.diag(s)

# <codecell>

n = 60
u1,s1,vt1 = U[:,0:n], S[0:n,0:n], Vt[0:n,:]
Xhat = sp.dot(sp.dot(u1,s1),vt1)
(u1.nbytes+s1.nbytes+vt1.nbytes) - X.nbytes

# <codecell>

plt.gray()
plt.imshow(Xhat)

# <codecell>

plt.imshow(X)

# <codecell>

((u1.nbytes+s1.nbytes+vt1.nbytes)*50000)*1e-9

# <codecell>

(u1.nbytes+s1.nbytes+vt1.nbytes)*1e-3

# <codecell>

(X.nbytes*50000)*1e-9

# <codecell>

X.nbytes*1e-6

# <headingcell level=5>

# The above shows that to store all 50,000 images before the SVD Algorithm we would need 80.4 Gb of hard disk space (each file taking 1.607 Mb). After the algorithm you would only need 23.088 Gb of hard disk save the database (with each file being only 461.76 kB)

