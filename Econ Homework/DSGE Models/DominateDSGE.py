import scipy as sp
from scipy import linalg as la
import sympy as sym
from sympy import symbols, nsolve, Symbol
import math
import numpy as np
from numpy.matlib import repmat
from RBCnumerderiv import numericalDerivatives

## Steps
#1 Find necessary conditions
#2 log linearize these conditions and constraints
#3 potulate a linear law of motion
#4 solve quadratic equations
#5 analyze model
#6 compared method of undetermined coefficients to a state space

## execfile('/Users/spencerlyon2/Dropbox/Python Labs/Spencer Lyon/Econ Homework/DSGE Models/DominateDSGE.py')
## execfile('/Users/spencerlyon/Dropbox/Python Labs/Spencer Lyon/Econ Homework/DSGE Models/DominateDSGE.py')
#define params
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

#use nsolve to solve for values#
SOLSS = nsolve((f1,f2,f3,f4,f5,f6),(Css, rss, lss, wss, Tss, kss), (.75,.12,.55,1.32,.041,4.05))

cs = float(SOLSS[0])
rs = float(SOLSS[1])
ls = float(SOLSS[2])
ws = float(SOLSS[3])
Ts = float(SOLSS[4])
ks = float(SOLSS[5])

solVec = sp.array([[ks,ls],[ks,ls],[ks,ls],[zbar],[zbar]])

theta0 = np.mat([ks,ls,ks,ls,ks,ls,zbar,zbar])

#define vars t_1 = t-1
ct, ct1 = symbols('ct, ct1')
rt, rt1 = symbols('rt, rt1')
wt = Symbol('wt')
lt, lt1, lt_1 = symbols('lt, lt1, lt_1')
Tt = Symbol('Tt')
kt, kt1, kt2 = symbols('kt, kt1, kt2')
zt1, zt = symbols('zt1, zt')

#Define State vars
xt1 = [kt2]
xt = [kt1]
xt_1 = [kt]

#Definition equations
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
	

#define characteristic equations

def dstuff(kt2, kt1, kt, lt1, lt, lt_1, zt1, zt): #change so kt2 and lt1 are together (xt1...)
	# k2t = xt1[0] ... lt1 = xt1[1]
	# repeat to unpack other xt, yt, zt vectors.
	rt = getrt(kt, lt, zt)
	wt = getwt(kt, zt, lt)
	Tt = getTt(wt, lt, rt, kt)
	ct = getct(wt, lt, rt, kt, Tt, kt1)
	rt1 = getrt1(kt1, lt1, zt1)
	wt1 = getwt1(kt1, zt1, lt1)
	Tt1 = getTt1(wt1, lt1, rt1, kt1)
	ct1 = getct1(wt1, lt1, rt1, kt1, Tt1, kt2)
	Euler1 = 1/(ct**gamma) - beta*((1/ct1**gamma)*((rt - delta)*(1 - tao) + 1))
	Euler2 = a*(1/((1-lt)**xsi)) - (1/ct**gamma) * wt * (1-tao)
	return Euler1, Euler2

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
    rt1 = getrt1(kt1, lt1, zt1)
    wt1 = getwt1(kt1, zt1, lt1)
    Tt1 = getTt1(wt1, lt1, rt1, kt1)
    ct1 = getct1(wt1, lt1, rt1, kt1, Tt1, kt2)
    #Euler2 = 1/(ct**gamma) - beta*((1/ct1**gamma)*((rt - delta)*(1 - tao) + 1))
    #Euler1 = a*(1/((1-lt)**xsi)) - (1/ct**gamma) * wt * (1-tao)
    Euler1 = wt*( 1-tao) * (1-lt)**xsi/(a*ct**gamma) - 1
    Euler2 = beta*(((ct/ct1)**gamma)*((rt1 - delta)*(1-tao) + 1)) - 1
    return [Euler1, Euler2]


F = sp.array([[dstuff(ks + epsilon, ks, ks, ls, ls, ls, zbar, zbar)[0]/-epsilon, dstuff(ks + epsilon, ks, ks, ls, ls, ls, zbar, zbar)[1]/-epsilon],[dstuff(ks, ks, ks, ls+ epsilon, ls, ls, zbar, zbar)[0]/-epsilon, dstuff(ks, ks, ks, ls + epsilon, ls, ls, zbar, zbar)[1]/-epsilon]])
G = sp.array([[dstuff(ks, ks + epsilon, ks, ls, ls, ls, zbar, zbar)[0]/-epsilon, dstuff(ks, ks + epsilon, ks, ls, ls, ls, zbar, zbar)[1]/-epsilon],[dstuff(ks, ks, ks, ls, ls+ epsilon, ls, zbar, zbar)[0]/-epsilon, dstuff(ks, ks, ks, ls, ls + epsilon, ls, zbar, zbar)[1]/-epsilon]])
H = sp.array([[dstuff(ks, ks, ks + epsilon, ls, ls, ls, zbar, zbar)[0]/-epsilon, dstuff(ks, ks, ks + epsilon, ls, ls, ls, zbar, zbar)[1]/-epsilon], [dstuff(ks, ks, ks, ls, ls, ls+ epsilon, zbar, zbar)[0]/-epsilon, dstuff(ks, ks, ks, ls, ls, ls+ epsilon, zbar, zbar)[1]/-epsilon]])
L = sp.array([[dstuff(ks, ks, ks, ls, ls, ls, zbar + epsilon, zbar)[0]/-epsilon, dstuff(ks, ks, ks, ls, ls, ls, zbar + epsilon, zbar)[1]/-epsilon]])
M = sp.array([[dstuff(ks, ks, ks, ls, ls, ls, zbar, zbar + epsilon)[0]/-epsilon, dstuff(ks, ks, ks, ls, ls, ls, zbar, zbar + epsilon)[1]/-epsilon]])


#print 'F', F
#print 'G', G
#print 'H', H
#print 'L', L
#print 'M', M

print "Done with Chase's Solving!"

A = sp.zeros((0,0))
B = sp.zeros((0,0))
C = sp.zeros((0,0))
D = sp.zeros((0,0))
J = sp.zeros((0,0))
K = sp.zeros((0,0))
N = sp.eye(xsi)


AA,BB,CC,DD,FF,GG,HH,JJ,KK,LL,MM,TT  = numericalDerivatives(dstuff2, theta0, 2,0,1)
CC = sp.array([])
NN = sp.array([xsi])
print "Done using Kerk's Code"