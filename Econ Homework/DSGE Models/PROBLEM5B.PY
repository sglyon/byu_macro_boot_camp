# -*- coding: utf-8 -*-
# <nbformat>3</nbformat>

# <headingcell level=1>

# Spencer Lyon

# <headingcell level=3>

# DSGE Homework: Lab 5b

# <headingcell level=4>

# Import Statements

# <codecell>

import scipy as sp
import sympy as sym
from sympy import symbols, nsolve, Symbol

# <codecell>

sp.set_printoptions(linewidth=140, precision=4)

# <headingcell level=4>

# Parameter Definitions

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

# <codecell>

C, r, L, w, T, k, y, i = symbols('C, r, L, w, T, k, y, i')
variables        = (k, w, r, L, C, T,y,i)
startingValues   = (4.05, 1.32, .12, .55, .75, .041, .5, .5)

# <headingcell level=4>

# Equation Definitions and Solution

# <codecell>

f1 = C - (1.0-tao)*(w*L + (r-delta)*k)-T
f2 = 1.0  - beta*((r-delta)*(1.0-tao)+1.0) 
f3 = a*(1.0-L)**(-xsi) - C**(-gamma)*w*(1.0-tao)
f4 = r - alpha*(L*expZbar/k)**(1-alpha)
f5 = w - (1.0- alpha)*expZbar*(k/(L*expZbar))**(alpha)
f6 = T - tao*(w*L + (r-delta)*k)
f7 = y - (k**alpha) - (L**(1-alpha))
f8 = i - delta*k  
functions = (f1, f2, f3, f4, f5, f6, f7, f8)

# <codecell>

xbar = sp.array(nsolve(functions, variables, startingValues),float); xbar

