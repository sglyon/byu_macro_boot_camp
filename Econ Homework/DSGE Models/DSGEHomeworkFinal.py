import scipy as sp
from scipy import linalg as la
import sympy as sym
from sympy import symbols, nsolve, Symbol
import math
import numpy as np
from numpy.matlib import repmat
from RBCnumerderiv import numericalDerivatives
from UhligSolve import solvingStuffForPQRS, nullSpaceBasis
import matplotlib.pyplot as plt
from random import gauss
from scipy import mean
from MonteCarloSimulation import Xgen, Ygen, Zgen, MonteCarlo
from ImpulseResponseFunctions import impulseResponse, ZgenImpulse

global gamma, xsi, beta, alpha, a, delta, zbar, tao, expZbar, epsilon

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
rho = 0.90


#Solve SS
Css = Symbol('Css')
rss = Symbol('rss')
lss = Symbol('lss')
wss = Symbol('wss')
Tss = Symbol('Tss')
kss = Symbol('kss')

#-----------------------------------------------------------------#
#------------------Variables/Equations----------------------------#
#-----------------------------------------------------------------#
f1 = Css - (1.-tao)*(wss*lss + (rss-delta)*kss) - Tss
f2 = 1. - beta*((rss-delta)*(1. - tao) +1)
f3 = (a/((1.-lss)**xsi))-((1./Css**gamma)*wss*(1.-tao))
f4 = rss - ((alpha)*(kss**(alpha-1.))*(lss**(1.-alpha)))
f5 = wss - ((1. - alpha)*(kss**alpha)*(lss**(-alpha)))
f6 = Tss - (tao*(wss*lss + (rss - delta)*kss))

# use nsolve to solve for SS values
SOLSS = nsolve((f1,f2,f3,f4,f5,f6),(Css, rss, lss, wss, Tss, kss), (.75,.12,.55,1.32,.041,4.05))

cs = float(SOLSS[0])
rs = float(SOLSS[1])
ls = float(SOLSS[2])
ws = float(SOLSS[3])
Ts = float(SOLSS[4])
ks = float(SOLSS[5])

theta0 = np.mat([ks,ls,ks,ls,ks,ls,zbar,zbar])

#D efinition equations
global getct, getrt, getwt, getTt, getrt1, getTt1, getw1, getct1

def getct(wt, lt, rt, kt, Tt, kt1):
	return(1.-tao)*(wt*lt + (rt - delta)*kt) + kt + Tt - kt1
def getrt (kt, lt, zt):
	return(alpha * (kt**(alpha-1.) *(lt*(math.e**(zt)))**(1.-alpha)))
def getwt(kt, zt, lt):
	return(1.-alpha) * (((kt/lt))**alpha) * ((math.e**zt)**(1.-alpha))
def getTt(wt, lt, rt, kt):
	return(tao)*(wt*lt + (rt - delta)*kt)
def getrt1(kt1, lt1, zt1):
	return alpha * (kt1**(alpha-1.) *(lt1*(math.e**(zt1)))**(1.-alpha))
def getTt1(wt1, lt1, rt1, kt1):
	return(tao)*(wt1*lt1 + (rt1 - delta)*kt1)
def getwt1(kt1, zt1, lt1):
	return(1.-alpha) * (kt1**alpha) * ((math.e**(zt1))**(1.-alpha)) * lt1**(-alpha)
def getct1(wt1, lt1, rt1, kt1, Tt1, kt2):
	return(1.-tao)*(wt1*lt1 + (rt1 - delta)*kt1) + kt1 + Tt1 - kt2


global dstuff2

def dstuff2(vals):
    kt2 = vals[0,0]
    lt1 = vals[0,1]
    kt1 = vals[0,2]
    lt = vals[0,3]
    kt = vals[0,4]
    lt_1 = vals[0,5]
    zt1 = vals[0,6]
    zt = vals[0,7]
    rt = getrt(kt, lt, zt)
    wt = getwt(kt, zt, lt)
    Tt = getTt(wt, lt, rt, kt)
    ct = getct(wt, lt, rt, kt, Tt, kt1)
    Tt1 = getTt1(wt1, lt1, rt1, kt1)
    ct1 = getct1(wt1, lt1, rt1, kt1, Tt1, kt2)
    rt1 = getrt1(kt1, lt1, zt1)
    wt1 = getwt1(kt1, zt1, lt1)
    Euler1 = wt*( 1-tao) * (1-lt)**xsi/(a*ct**gamma) - 1
    Euler2 = beta*(((ct/ct1)**gamma)*((rt1 - delta)*(1-tao) + 1)) - 1
    return sp.array([Euler1, Euler2])

global AA, BB, CC, DD, FF, GG, HH, JJ, KK, LL, MM, TT, NN

AA,BB,CC,DD,FF,GG,HH,JJ,KK,LL,MM,TT  = numericalDerivatives(dstuff2, theta0, 2,0,1)
NN = sp.array([rho])

print "Done!!!!!!!!!!!!!!!!!!!!!!!!"

print 'Trying to solve for PQRS'

global PP, QQ, RR, SS
PP,QQ,RR,SS = solvingStuffForPQRS(AA, BB, CC, DD, FF, GG, HH, JJ, KK, LL, MM, NN)
print "It freakin' worked. We have P, Q, R, S"

#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
#---------------------------From Timothy and Sara------------------------------#

#Timothy Hills: Y I C generator

global YICgen
def YICgen(X,Z,alpha, delta):
	"""
	Yt = (Kt**alpha)*((exp(Zt)*Lt)**(1-alpha))
	This function generates the output levels
	given the previously defined X = sp.array([[K],[L]])
	(X is a 2xT period matrix of capital on top and
	labor on the bottom row) and Z (previously generated
	technology shocks).
	It = Ktp1 - (1 - delta)*Kt
	This function generates the investment levels per period
	delta = depreciation rate of capital.
	Ct = Yt - It
	This function defines the consumption levels as a
	difference between output and investment.
	"""
	K = X[0,:]
	L = X[1,:]
	t = sp.shape(X)[1]
	Y = sp.zeros(t)
	I = sp.zeros(t)
	C = sp.zeros(t)
	#solve for Y in each period t
	for i in range(t):
		Y[i] = (K[i]**alpha)*((sp.exp(Z[i])*L[i])**(1.-alpha))
	#solve for I in each period t
	for i in range(t-1):
		I[i] = K[i+1] - (1. - delta)*K[i]
	#solve for C in each period t
	for i in range(t-1):
		C[i] = Y[i] - I[i]
	return Y, I, C

# Setup for Problems 13-14
#X0=sp.array([[-.1,0.]])
global X0, Z0, Y0, Xbar, Zbar, Mu, Var, reps, T_monte
global shock, T_irf

X0_m = sp.array([[-.1,0.]])
Z0 = -math.sqrt(.004)
Y0 = 0
Xbar = sp.mat([[ks,ls]])
Zbar = 0
Mu = 0
Var = .0004
reps = 1000
T_monte = 250
GDP_monte,I_monte,C_monte= MonteCarlo(PP,QQ,RR,SS,NN,X0_m,Y0,Z0,Xbar,Zbar,Mu,
                                      Var,alpha,delta,reps,T_monte, YICgen)



X0_irf = sp.array([[0,0]])
shock = -.005
T_irf = 40

GDP_irf, invest_irf, consumption_irf=impulseResponse(PP, QQ, RR, SS, NN, X0_irf,
                                                     Y0, YICgen, shock, Xbar,
                                                     Zbar, alpha, delta,
                                                     T_irf)
