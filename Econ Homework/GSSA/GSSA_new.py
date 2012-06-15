"""
Created Ju, 2012

Author: Everyone
"""

import scipy as sp
from scipy import linalg as la
import sympy as sym
from sympy import symbols, nsolve, Symbol
from scipy.integrate import quadrature as quad
from scipy import stats as stat
import matplotlib.pyplot as plt
import math
import numpy as np
import SolveSS as sols

##------------------------OLD CODE FROM DSGE ASSIGNMENT-----------------------##
# Model equations
def getct(wt, lt, rt, kt, Tt, kt1):
	return(1.-tao)*(wt*lt + (rt - delta)*kt) + kt + Tt - kt1
def getrt (kt, lt, zt):
	return(alpha * (kt**(alpha-1.) *(lt*(math.e**(zt)))**(1.-alpha)))
def getwt(kt, zt, lt):
	return(1.-alpha) * (((kt/lt))**alpha) * ((math.e**zt)**(1.-alpha))
def getTt(wt, lt, rt, kt):
	return(tao)*(wt*lt + (rt - delta)*kt)
def getrt1(kt1, lt1, zt1):
	return alpha * ((1./kt1)**(1.-alpha) *(lt1*(math.e**(zt1)))**(1.-alpha))
def getTt1(wt1, lt1, rt1, kt1):
	return(tao)*(wt1*lt1 + (rt1 - delta)*kt1)
def getwt1(kt1, zt1, lt1):
	return(1.-alpha) * (kt1**alpha) * ((math.e**(zt1))**(1.-alpha)) * lt1**(-alpha)
def getct1(wt1, lt1, rt1, kt1, Tt1, kt2):
	return(1.-tao)*(wt1*lt1 + (rt1 - delta)*kt1) + kt1 + Tt1 - kt2

# parameters
gamma = 2.5
xsi = 1.5
a = .5
cs,rs,ls,ws,Ts,ks = sols.solveSS(gamma, xsi, a)

##---------------------------------NEW STUFF----------------------------------##
#funnystuff
def fun(iter):
	compliment = sp.array(["you aren't that far off!!","you are awesome!","your sister is hott!","nice shirt","you'd be top of your class if you switched to philosophy.  think about it.","a dog is forever in the pushup position.","you aren't the lowest hanging fruit.","i like your face.","you are cute."])
	n = sp.random.randint(0,9)
	m = sp.random.randint(1,3)
	if iter %m == 0:
		print compliment[n]

eta = .1
J = 6
junk_1 = .05
junk_2 = .07
junk_3 = .01
junk_4 = .09
junk_5 = .08
junk_6 = .07
lt = 1
lt1 = 1

# Choose simulation length,T. Draw a random time series of innovations
T = 10000
sigma = np.sqrt(0.004)
beta  = 0.98
alpha = 0.4
delta = 0.10
zbar  = 0.
tao   = 0.05
expZbar = 1.
epsilon = .1
rho = 0.90
epsi10000 = sp.random.randn(T+2)
epsi = epsi10000*sigma

# Use law of motion to generate time series of technology levels
z = []
z.append(zbar)
for i in range(1,T+2):
	z.append((1 - rho)*zbar + rho*z[i-1]+epsi[i])

# Initial guess for b
bnew = sp.vstack([junk_1, junk_2, junk_3, junk_4, junk_5, junk_6])
f = lambda x: bnew[0] + bnew[1]*x[0] + bnew[2]*x[1] + bnew[3]*x[0]**2 +\
              bnew[4]*x[1]**2 + bnew[5]*x[0]*x[1]

# Iterate using b to generate the model for T periods
k = []
k.append(ks)
for i in range(1,T+2):
	x = []
	x.append(k[i-1])
	x.append(z[i])
	k.append(sp.absolute(float(f(x))))


# Quadrature and simulation
difference = 1e+15
iter = 0
DIFF = []

ones = sp.vstack(sp.ones(T))
kcol = sp.vstack([k[0:T]]).T
zcol = sp.vstack([z[0:T]]).T
ksq = sp.vstack([kcol**2])
zsq = sp.vstack([zcol**2])
kz = sp.vstack([kcol*zcol])
A = sp.hstack([ones,kcol,zcol,ksq,zsq,kz])

U,s,V = la.svd(A,full_matrices=False)
S = la.diagsvd(s,U.shape[1],V.shape[1])


while (difference > .01) and (iter < 150):
	iter += 1
	print iter
	fun(iter)
	bold = bnew
	KAPPA = []
	KAPPA.append(k[0])
	#print k
	for i in range(1,T):
		projeps = sp.random.randn(J)*sigma # projected epsilons for the
		projeps.sort()
		omega = []
		for j in range(J):
			if(projeps[j]<=0):
				omega.append(stat.norm.cdf(projeps[j]))
			else:
				omega.append(1-stat.norm.cdf(projeps[j]))

		rt = getrt(k[i], lt, z[i])
		wt = getwt(k[i], z[i], lt)
		Tt = getTt(wt, lt, rt, k[i])
		ct = getct(wt, lt, rt, k[i], Tt, k[i+1])
		rt1 = getrt1(k[i+1], lt1, z[i+1])
		wt1 = getwt1(k[i+1], z[i+1], lt1)
		Tt1 = getTt1(wt1, lt1, rt1, k[i+1])
		ct1 = getct1(wt1, lt1, rt1, k[i+1], Tt1, k[i+2])
		#print 'rt is', rt, 'wt is', wt, 'Tt is', Tt
		#print 'ct is ', ct, 'ct+1 is ', ct1

		kappa = 0
		for j in range(J):
			kappa += (beta*omega[j]*((ct/ct1)**gamma)*(1.+rt1-delta)*k[j+1])
		KAPPA.append(kappa)
	# Update guess for parameters,b, by minimizing errors in the non-linear regression

	#print KAPPA

	#print A

	bprime = sp.dot(sp.dot(V.T,sp.dot(la.inv(S),U.T)),KAPPA)
	bnew = []
	if(iter == 1):
		bnew = ((1.-eta)*bold + eta*bprime)[0]
	else:
		bnew = (1.-eta)*bold + eta*bprime
	#print bnew
	difference = la.norm(bnew-bold)
	DIFF.append(difference)
	print difference

# Check to see if the convergence critera is met, i.e., old b - new b is small.
# if not make a convex combination of the of old and new values return to step 5

# else: construct and evaluate the Euler equation errors

KAPPAarray = np.array(KAPPA)
KAPPArows = int(KAPPAarray.shape[0])
kapOnes = sp.vstack(sp.ones((KAPPArows,1)))
#print KAPPA, 'KAPPA'
#eulerError = la.norm(LAST-kapOnes)#KAPPA - kcol
xAxis = sp.arange(0,T,1)
plt.figure()
plt.plot(xAxis[20:],kcol[20:], '*')
plt.plot(xAxis[20:], KAPPA[20:],'o')
plt.show()
plt.figure()
DIFFarray = np.array(DIFF)
DIFFlength = int(DIFFarray.shape[0])
plt.plot(xAxis[4:DIFFlength],DIFF[4:])
plt.show()
#print eulerError
print 'new b', bnew
#print 'original b', borig
