# -*- coding: utf-8 -*-
# <nbformat>3</nbformat>

# <codecell>

import scipy as sp
import sympy as sym
from sympy import Symbol, nsolve, symbols

# <codecell>

tao = 0.05
gamma = 2.5
beta = 0.98
alpha = 0.40
delta = 0.10
z = 0

# <codecell>

C, r, l, w, T, k, y, i = symbols('C, r, l, w, T, k, y, i')
startingValues = (0.75,0.12,1.32,0.41,4.05,.1,.1)

# <codecell>

f1 = C - (1.0- tao) * (w +(r - delta)*k) - T 
f2 = 1.0 - beta*((r-delta)*(1-tao)+1.0)
f3 = r - alpha*k**(alpha-1)
f4 = w - (1- alpha)*k**(alpha)
f5 = T - tao*(w+ (r-delta)*k)
f6 = y - (k**alpha)
f7 = i - delta*k  

# <codecell>

nsolve((f1,f2,f3,f4,f5,f6,f7),(r, k, w, T, C,y,i),startingValues)

# <codecell>


