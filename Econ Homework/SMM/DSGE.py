import scipy as sp
from scipy import linalg as la
import sympy as sym
from sympy import symbols, nsolve, Symbol
import math
import numpy as np
from numpy.matlib import repmat
from NumDeriv import numericalDerivatives
from UhligSolve import solvingStuffForPQRS, nullSpaceBasis
import matplotlib.pyplot as plt
from random import gauss
from scipy import mean
from MonteCarloSimulation import MonteCarlo #Use to have Xgen,Zgen,Ygen
from NumDeriv import numericalDerivatives
from SolveSS import solveSS

def DSGE_sim(beta1,T,Z):
	beta  = 0.98
	alpha = 0.4
	delta = 0.10
	zbar  = 0.
	tao   = 0.05
	expZbar = 1.
	epsilon = .1
	rho = 0.90

	gamma, xsi, a = beta1

	cs,rs,ls,ws,Ts,ks = solveSS(gamma,xsi,a)

	#print 'Found Steady States'

	theta0 = np.mat([ks,ls,ks,ls,ks,ls,zbar,zbar])

	AA,BB,CC,DD,FF,GG,HH,JJ,KK,LL,MM,TT  = numericalDerivatives(theta0, 2,0,1,xsi,gamma,a)
	NN = sp.array([rho])

	#print "Found Numerical Derivatives"

	P,Q,R,S = solvingStuffForPQRS(AA, BB, CC, DD, FF, GG, HH, JJ, KK, LL, MM, NN)
	#print "We have P, Q, R, S"


	X0_m = sp.array([[-.1,0.]])
	Xbar = sp.mat([[ks,ls]])
	Zbar = 0
	reps = 10000
	C_monte,L_monte,W_monte,C_var_monte= MonteCarlo(P,Q,R,S,NN,X0_m,Z,Xbar,alpha,delta,reps,T)
	return C_monte,L_monte,W_monte,C_var_monte
