# -*- coding: utf-8 -*-
# <nbformat>3</nbformat>

# <codecell>

from numpy import *
from scipy.optimize import fsolve

# <codecell>

# Define Parameters
beta    = 0.96**20;
delta   = 1 - (1-0.05)**20;
gamma   = 3;
A       = 1;
alpha   = 0.35;
rho     = .5;

# <codecell>

# Define functions
uPrime = lambda c: 1/c**gamma
c1     = lambda wt, k2tP1: wt - k2tP1
c2tP1  = lambda wtP1, rtP1, k2tP1, k3tP1: wtP1 + (1 + rtP1 - delta) *k2tP1 - k3tP1;
c2tP2  = lambda rtP2, k3tP2: (1 + rtP2 - delta) * k3tP2
wt     = lambda k2t, k3t: (1-alpha) * A * 2**-alpha*(k2t+k3t)**alpha
rt     = lambda k2t, k3t: alpha * A * (2**(1-alpha)) * (k2t + k3t)**(alpha-1)

# <codecell>

kinit = array([.1,.1]);

# <codecell>

fsolve?

# <codecell>


