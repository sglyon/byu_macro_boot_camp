# -*- coding: utf-8 -*-
# <nbformat>3</nbformat>

# <headingcell level=1>

# Spencer Lyon

# <headingcell level=3>

# DSGE Homework Lab7

# <headingcell level=4>

# Import Statements

# <codecell>

import scipy as sp
import sympy as sym
from sympy import symbols, nsolve, Symbol

# <headingcell level=4>

# Find the steady state values

# <codecell>

gamma = 2.5
xsi= 1.5
beta  = 0.98
alpha = 0.4
a     = 0.5
delta = 0.10
zbar  = 0
tao   = 0.05
expZbar = sp.exp(zbar)
epsilon = .1

# <codecell>

C, r, L, w, T, k = symbols('C, r, L, w, T, k')
variables        = (k, w, r, L, C, T)
startingValues   = (4.05, 1.32, .12, .55, .75, .041)

# <codecell>

f1 = C - (1.0-tao)*(w*L + (r-delta)*k)-T
f2 = 1.0  - beta*((r-delta)*(1.0-tao)+1.0) 
f3 = a*(1.0-L)**(-xsi) - C**(-gamma)*w*(1.0-tao)
f4 = r - alpha*(L*expZbar/k)**(1-alpha)
f5 = w - (1.0- alpha)*expZbar*(k/(L*expZbar))**(alpha)
f6 = T - tao*(w*L + (r-delta)*k)
functions = (f1, f2, f3, f4, f5, f6)

# <codecell>

xbar = sp.array(nsolve(functions, variables, startingValues)[0:4],float); xbar

# <codecell>

xbar

# <headingcell level=4>

# Numerical derivative with respect to delta

# <codecell>

epsilon = .1
gamma = 2.5
xsi= 1.5
beta  = 0.98
alpha = 0.4
a     = 0.5
delta = 0.10 + epsilon
zbar  = 0
tao   = 0.05
expZbar = sp.exp(zbar)
f1 = C - (1.0-tao)*(w*L + (r-delta)*k)-T
f2 = 1.0  - beta*((r-delta)*(1.0-tao)+1.0) 
f3 = a*(1.0-L)**(-xsi) - C**(-gamma)*w*(1.0-tao)
f4 = r - alpha*(L*expZbar/k)**(1-alpha)
f5 = w - (1.0- alpha)*expZbar*(k/(L*expZbar))**(alpha)
f6 = T - tao*(w*L + (r-delta)*k)
functions = (f1, f2, f3, f4, f5, f6)
xbardelta = sp.array(nsolve(functions, variables, startingValues)[0:4],float)

# <codecell>

dxbardelta = (xbardelta - xbar)/epsilon; dxbardelta

# <headingcell level=4>

# Numerical derivative with respect to tao

# <codecell>

epsilon = .1
gamma = 2.5
xsi= 1.5
beta  = 0.98
alpha = 0.4
a     = 0.5
delta = 0.10
zbar  = 0
tao   = 0.05 + epsilon
expZbar = sp.exp(zbar)
f1 = C - (1.0-tao)*(w*L + (r-delta)*k)-T
f2 = 1.0  - beta*((r-delta)*(1.0-tao)+1.0) 
f3 = a*(1.0-L)**(-xsi) - C**(-gamma)*w*(1.0-tao)
f4 = r - alpha*(L*expZbar/k)**(1-alpha)
f5 = w - (1.0- alpha)*expZbar*(k/(L*expZbar))**(alpha)
f6 = T - tao*(w*L + (r-delta)*k)
functions = (f1, f2, f3, f4, f5, f6)
xbartao = sp.array(nsolve(functions, variables, startingValues)[0:4],float)

# <codecell>

dxbartao = (xbartao - xbar)/epsilon; dxbartao

# <headingcell level=4>

# Numerical derivative with respect to zbar

# <codecell>

epsilon = .1
gamma = 2.5
xsi= 1.5
beta  = 0.98
alpha = 0.4
a     = 0.5
delta = 0.10 
zbar  = 0 + epsilon
tao   = 0.05
expZbar = sp.exp(zbar)
f1 = C - (1.0-tao)*(w*L + (r-delta)*k)-T
f2 = 1.0  - beta*((r-delta)*(1.0-tao)+1.0) 
f3 = a*(1.0-L)**(-xsi) - C**(-gamma)*w*(1.0-tao)
f4 = r - alpha*(L*expZbar/k)**(1-alpha)
f5 = w - (1.0- alpha)*expZbar*(k/(L*expZbar))**(alpha)
f6 = T - tao*(w*L + (r-delta)*k)
functions = (f1, f2, f3, f4, f5, f6)
xbarzbar = sp.array(nsolve(functions, variables, startingValues)[0:4],float)

# <codecell>

dxbarzbar = (xbarzbar - xbar)/epsilon; dxbarzbar

# <codecell>


