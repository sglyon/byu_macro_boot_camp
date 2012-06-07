# -*- coding: utf-8 -*-
# <nbformat>3</nbformat>

# <codecell>

from scipy import eye
import scipy as sp
def rowswap(n,j,k):
    
    out = eye(n)
    out[j-1,j-1] = 0
    out[k-1,k-1] = 0
    out[j-1,k-1] = 1
    out[k-1,j-1]= 1
    return sp.mat(out)

def cmult(n,j,const):
    out = eye(n)
    out[j-1,j-1] = const
    return sp.mat(out)

def cmultadd(n,k,j,const):
	out = eye(n)
	out[j-1,k-1] = const
	return sp.mat(out)

# <codecell>


