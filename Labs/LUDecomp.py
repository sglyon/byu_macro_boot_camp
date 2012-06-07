# -*- coding: utf-8 -*-
# <nbformat>3</nbformat>

# <codecell>

import scipy as sp



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


