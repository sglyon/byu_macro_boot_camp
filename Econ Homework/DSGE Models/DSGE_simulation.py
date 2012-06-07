# -*- coding: utf-8 -*-
# <nbformat>2</nbformat>

# <codecell>

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

theta0 = np.mat([ks,ls,ks,ls,ks,ls,zbar,zbar])

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
    Euler1 = wt*( 1-tao) * (1-lt)**xsi/(a*ct**gamma) - 1
    Euler2 = beta*(((ct/ct1)**gamma)*((rt1 - delta)*(1-tao) + 1)) - 1
    return sp.array([Euler1, Euler2])

AA,BB,CC,DD,FF,GG,HH,JJ,KK,LL,MM,TT  = numericalDerivatives(dstuff2, theta0, 2,0,1)
CC = sp.array([])
NN = sp.array([[.9]])
print "Done!!!!!!!!!!!!!!!!!!!!!!!!"

# <codecell>

import scipy as sp
from scipy import linalg as la
import numpy as np
from numpy import linalg as npla
import matplotlib.pyplot as plt


def mparray2npfloat(a):
    """convert a NumPy array of mpmath objects (mpf/mpc)
    to a NumPy array of float"""
    tmp = a.ravel()
    res = _cfunc_float(tmp)
    return res.reshape(a.shape)

TOL = .000001


def nullSpaceBasis(A):
    """
    Finds the nullspace of the matrix A using the SVD.
    Returns basis for nullspace.
    """
    U,s, Vh = la.svd(A)
    vecs = np.array([])
    toAppend = A.shape[1] -s.size
    s = sp.append(s,sp.zeros((1,toAppend)))
    counter = 0
    for i in range(0,s.size):
        if s[i]==0:
            vecs = Vh[-toAppend:,:]
    if vecs.size ==0:
        vecs = sp.zeros((1,A.shape[1]))
    return sp.mat(vecs)
	
def solve_P(F,G,H):
    """This function takes arguments for F,G,H and solves the matrix quadratic given by
    F*P^2+G*P+H=0.  Note F, G, and H must be square.
    The function returns the matrix P and the resulting matrix, given by F*P^2+G*P+H 
    which should be close to zero.
    The algorithm used to solve for P is outlined in 'A Toolkit for Analyzing Nonlinear 
    Dynamic Stochastic Models Easily' by Harald Uhlig.
    """
    m=sp.shape(F)[0]
    
    Xi=sp.concatenate((-G,-H), axis=1)
    second=sp.concatenate((sp.eye(m,m),sp.zeros((m,m))),axis=1)
    Xi=sp.concatenate((Xi,second))
    
    Delta=sp.concatenate((F,sp.zeros((m,m))),axis=1)
    second=sp.concatenate((sp.zeros((m,m)),sp.eye(m,m)),axis=1)
    Delta=sp.concatenate((Delta,second))
    
    (L,V) = la.eig(Xi,Delta)
    
    boolean = sp.zeros(len(L))
    trueCount =0
    
    for i in range(len(L)):
        if L[i]<1 and L[i]>-1 and sp.imag(L[i])==0 and trueCount<m:
            boolean[i] = True
            trueCount+=1
    #display(L, boolean)
    if trueCount<m:
        print "Imaginary eigenvalues being used"
        for i in range(len(L)):
            if math.sqrt(real(L[i])**2+imag(L[i])**2)<1 and trueCount<m:
                boolean[i]=True
                trueCount+=1
    #display(boolean)
    
    if trueCount==m:
        print "true count is m"
        Omega=sp.zeros((m,m))
        diagonal=[]
        count =0
        for i in range(len(L)):
            if boolean[i]==1:
                Omega[:,count]=sp.real(V[m:2*m,i])+sp.imag(V[m:2*m,i])
                diagonal.append(L[i])
                count+=1
        Lambda=sp.diag(diagonal)
        try:
            P=sp.dot(sp.dot(Omega,Lambda),la.inv(Omega))
            #for j in range(sp.shape(P)[0]):
                #for k in range(sp.shape(P)[1]):
                    #if abs(P[j,k])<1e-5:
                        #P[j,k]=0
        except:
            print 'Omega not invertable'
            P=sp.zeros((m,m))
        diff=sp.dot(F,sp.dot(P,P))+sp.dot(G,P)+H
        return P,diff
    else:
        print "Problem with input, not enough 'good' eigenvalues"
        return sp.zeros((m,m)),sp.ones((m,m))*100
		
def solvingStuffForPQRS(AA, BB, CC, DD, FF, GG, HH, JJ, KK, LL, MM, NN):
    """
    Solves for P,Q,R, and S in Uhlig's paper on DSGE models.

    P is found by solving the Matrix quadratic question.  Note that 
    solutions may not be unique. 
    Special Case:  FP^2+GP+H=0
    General Case:  ____________

    Q is found using Kronkier products.
    Special Case: FQN+(FP+G)Q+LN+M=0
    General Case: ___________________

    R,S are found only if Jump variables are included; otherwise, 
    they are returned as matrices of size 0 by ________.
    R = -inv(C)*(AP+B)
    S = -inv(C)*(AQ+D)

    Note: C is assumed to be of full rank and thus invertable
    Returns P,Q,R,S
    """
    q_eqns = sp.shape(FF)[0]
    m_states = sp.shape(FF)[1]
    l_equ = sp.shape(CC)[0]
    if l_equ==0:
        n_endog=0
    else:
        n_endog = sp.shape(CC)[1]
        
    k_exog = min(sp.shape(sp.mat(NN))[0], sp.shape(sp.mat(NN))[1])
    
    if l_equ==0:
        PP, Diff = solve_P(FF, GG, HH)
        PP = sp.real(PP)
    else:
        Cinv = la.inv(CC)
        Fprime=FF-sp.dot(sp.dot(JJ,Cinv),AA)
        Gprime=-(sp.dot(sp.dot(JJ,Cinv),BB)-GG+sp.dot(sp.dot(KK,Cinv),AA))
        Hprime=-sp.dot(sp.dot(KK,Cinv),BB)+HH
        PP, Diff = solve_P(Fprime, Gprime, Hprime)
        PP = sp.real(PP)

    #solve for QRS
    if l_equ ==0:
        RR = sp.zeros((0,m_states))
        Left = FF
        Right = sp.dot(FF,PP) + GG
        VV = (la.kron(NN.T, Left)) + la.kron(sp.eye(k_exog), Right)
    else:
        RR=sp.dot(-Cinv,(sp.dot(AA,PP)+BB))
        Left = FF-sp.dot(sp.dot(JJ,Cinv),AA)
        Right = sp.dot(JJ,RR) + sp.dot(FF,PP)+G-sp.dot(sp.dot(KK,Cinv),AA)
        VV = (la.kron(NN.T, Left)) + la.kron(sp.eye(k_exog), Right)
    if False and (npla.matrix_rank(VV) < k_exog*(m_states + n_endog)):
        print("Sorry but V is note invertible. Can't solve for Q and S")
    else:
        if l_equ==0:
            Vector = (sp.dot(LL,NN) + MM).flatten()
            QQSS_vec = sp.dot(la.inv(VV),Vector)
            QQSS_vec = -QQSS_vec
        else:
            Vector = (sp.dot((sp.dot(sp.dot(JJ,Cinv),DD)-LL),NN)+sp.dot(sp.dot(KK,Cinv),DD)-MM).flatten()
            QQSS_vec = sp.dot(la.inv(VV),Vector)
            QQSS_vec = -QQSS_vec
            
        QQ = sp.reshape(QQSS_vec,(m_states,k_exog))
        
        if l_equ==0:
            SS=sp.zeros((0,2))
        else:
            SS = sp.dot(-Cinv,(sp.dot(AA,QQ)+DD))
    return PP, QQ, RR, SS
    

# <codecell>

def Xgen(X0,Z,PP,QQ,Xbar):
    """
    This function generates a history of X given a history 
    technology shocks (Z), a P matrix, a Q matrix, and an 
    intial X (X0).  
    Note Xt(tilde) = PXt-1(tilde) +QZt(tilde)
    Xt=Xbar*e^Xt(tilde)
    """
    num_endog=sp.shape(PP)[1] 
    T=len(Z)#sp.shape(Z)[0] 
    #display(T)
    X=sp.zeros((num_endog,T))
    X[:,0]=X0
    for i in range(1,T):
        Zt=Z[i]
        Xt_1=sp.zeros((num_endog,1))
        for j in range(num_endog):
            Xt_1[j,0]=X[j,i-1]
        Xt=sp.dot(PP,Xt_1)+sp.dot(QQ,Zt)
        for k in range(num_endog):
            X[k,i]=Xt[k,0]
    exponents=sp.exp(X)
    for p in range(T):
        for q in range(num_endog):
            X[q,p]=Xbar[0,q]*exponents[q,p]
    return X

import scipy as sp
import math
from random import gauss

def Zgen(Initial, RHOz, MUz, VARz,zbar,t):
	"""
	Zt = RHOz*Ztm1 + EPSz
	This randomly generates the technology shocks given 
	a specified RHOz (=N) and EPSz i.i.d.~(MUz,VARz) for
	a specified number of periods t.
	Z0 = 0
	Z1 = Initial
	"""
	#Standard deviation of Z needed for future application (gauss)
	SIGz = math.sqrt(VARz)
	#Generate Z for t periods, we will populate each period below
	Z = sp.zeros(t)
	#The first period of Z we assume all technology shock is 0
	#The second period the Z = indicated initial value
	Z[0] = 0
	Z[1] = Initial
	#Zt = RHOz*Ztm1 + EPSz where EPSz i.i.d.~(MUz, VARz)
	for i in range(2,t):
		Z[i] = RHOz*Z[i-1] + gauss(MUz,SIGz)
        for j in range(t):
            Z[j]=Z[j]+zbar
	#Should give you what you want!
	return Z

def Ygen(Y0,Z,RR,SS):
    """
    This function generates a history of Y given a history 
    technology shocks (Z), a R matrix, a S matrix, and an 
    intial Y (Y0).  
    Note Yt = RXt-1 +SZt
    """
    num_exog=sp.shape(RR)[1] 
    T=sp.shape(Z)[1] 
    Y=sp.zeros((num_exog,T))
    Y[:,0]=Y0
    for i in range(1,T):
        Zt=Z[0,i]
        Yt_1=sp.zeros((num_exog,1))
        for j in range(num_exog):
            Yt_1[j,0]=Y[j,i-1]
        Yt=sp.dot(RR,Yt_1)+sp.dot(SS,Zt)
        for k in range(num_exog):
            Y[k,i]=Yt[k,0]
    return Y

#Timothy Hills
#Y I C generator

import scipy as sp
import math

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

# <codecell>

MM

# <codecell>

P,Q,R,S=solvingStuffForPQRS(AA, BB, CC, DD, FF, GG, HH, JJ, KK, LL, MM, NN)
diffP=sp.dot(sp.dot(FF,P),P)
diffQ=sp.dot(sp.dot(FF,Q),NN)+sp.dot((sp.dot(FF,P)+GG),Q)+sp.dot(LL,NN)+MM
display(diffP, diffQ)
display(P,Q)

# <codecell>

#solve for steady state - Chase, Spencer
#solve for P,Q,R,S

#Monte Carlo simulation

#moments simulation

#impulse function
def MonteCarlo(P,Q,R,S,N,X0,Y0,Z0,Xbar,Zbar,Mu,Var,alpha,delta,reps,T):
    """
    Runs Monte Carlo simulation for DSGE model.
    Inputs:
        P,Q,R,S,N : Matrices previously found from steady state and solving
        X0,Y0,Z0 : Intial values for X,Y,Z histories
        Xbar, Zbar: Steady State Values for X and Z
        Mu : Mean value of epsilon shocks
        Var : Variance of epsilon shocks
        alpha, delta : constants from the model
        reps : Number of simulations to run
        T : time periods to run the simulation for
    """
    GDP = sp.zeros((T,reps))
    invest = sp.zeros((T,reps))
    consumption = sp.zeros((T,reps))
    
    for i in range(reps):
        #generate X,Y,Z
        Z=[]
        X=[]
        Z = Zgen(Z0,N,Mu,Var,Zbar, T)
        X = Xgen(X0,Z,P,Q,Xbar)
        
        #find y,i,c for each time period
        GDP[:,i],invest[:,i],consumption[:,i] = YICgen(X,Z,alpha,delta)
        
        #Y = Ygen(Y0,Z,R,S)

    #return GDP,invest,consumption
    GDP_mean = sp.zeros((T-1))
    invest_mean = sp.zeros((T-1))
    consumption_mean =sp.zeros((T-1))
    for j in range(0,T-1): #don't use first and last periods
        GDP_mean[j] = mean(GDP[j,:])
        invest_mean[j] = mean(invest[j,:])
        consumption_mean[j] = mean(consumption[j,:])
    
    #return GDP_mean,invest_mean,consumption_mean
    
    #might need to chop off first and last periods
    GDP_sort=sp.sort(GDP)
    invest_sort=sp.sort(invest)
    consumption_sort=sp.sort(consumption)
    
    
    lowerBound=.05*float(reps)
    lowerBound= lowerBound-lowerBound%1 #5%
    
    upperBound=reps-lowerBound #95%
    #lowerBound=0
    #upperBound=4
    
    GDP_lower=GDP_sort[:,lowerBound]
    GDP_upper=GDP_sort[:,upperBound]
    invest_lower=invest_sort[:,lowerBound]
    invest_upper=invest_sort[:,upperBound]
    consumption_lower=consumption_sort[:,lowerBound]
    consumption_upper=consumption_sort[:,upperBound]
    
    #plot mean, and 90% confidence bands for GDP,I,C
    x = arange(0,T-1,1)
    plt.figure(1)
    ll = plt.plot(x,GDP_lower[:-1])
    mm = plt.plot(x,GDP_upper[:-1])
    aa = plt.plot(x,GDP_mean)
    plt.show()
    plt.figure(2)
    ll = plt.plot(x,invest_lower[:-1])
    mm = plt.plot(x,invest_upper[:-1])
    aa = plt.plot(x,invest_mean)
    plt.show()
    plt.figure(3)
    ll = plt.plot(x,consumption_lower[:-1])
    mm = plt.plot(x,consumption_upper[:-1])
    aa = plt.plot(x,consumption_mean)
    plt.show()
    
    #return GDP_mean,invest_mean,consumption_mean,GDP_lower,GDP_upper,invest_lower, invest_upper,consumption_lower, consumption_upper
    return GDP, invest, consumption
    #sort y,i,c
    #plot mean for y,i,c across all periods
    #plot 5 and 95 bands
    
    

# <codecell>

#solve for steady state - Chase, Spencer
#solve for P,Q,R,S

#Monte Carlo simulation

#moments simulation

#impulse function

import matplotlib.pyplot as plt
X0=sp.array([[-.1,0.]])
Z0=-math.sqrt(.004)
Y0=0
Xbar=sp.mat([[ks,ls]])
Zbar=0
Mu=0
Var=.0004
alpha=.4
delta=.1
reps=1000
T=250
GDP,I,C= MonteCarlo(P,Q,R,S,NN,X0,Y0,Z0,Xbar,Zbar,Mu,Var,alpha,delta,reps,T)

# <codecell>

def ZgenImpulse(shock, RHOz,zbar,t):
	"""
	Zt = RHOz*Ztm1 + impulse
	Generates impulse technology shocks given 
	a specified RHOz (=N) and shock size for
	a specified number of periods t.
	Z3 = shock
	"""
	#Generate Z for t periods, we will populate each period below
	Z = sp.zeros(t)
	#The third period the Z = shock
	Z[0] = 0
	Z[3] = shock
        #Z[T/2]=-shock
        for j in range(t):
            Z[j]=Z[j]+zbar
	return Z

# <codecell>

def impulseResponse(P,Q,R,S,N,X0,Y0,shock,Xbar,Zbar,alpha,delta,T):
    """
    Produces graphs for the impulse responses of GDP, Investments,
    and consumption.  P,Q,R,S,N are matrices that have previously
    been solved for.  Xbar and Zbar are from finding the steady 
    state.  Alpha, delta, and T (number of periods) are parameters 
    for the model.  The variable 'shock' is the amount of shock 
    delivered to the model in the 4th time period.  

    Returns vectors and plots for GDP, investments, and consumption.
    """
    GDP = sp.zeros((T))
    invest = sp.zeros((T))
    consumption = sp.zeros((T))
    
    #generate X,Y,Z
    Z = ZgenImpulse(shock, N,Zbar,T)
    X = Xgen(X0,Z,P,Q,Xbar)
        
    #find y,i,c for each time period
    GDP,invest,consumption = YICgen(X,Z,alpha,delta)
        
    #Y = Ygen(Y0,Z,R,S)
    
    #return GDP_mean,invest_mean,consumption_mean  
    
    #plot mean, and 90% confidence bands for GDP,I,C
    x = arange(0,T-1,1)
    plt.figure(1)
    ll = plt.plot(x,GDP[:-1])
    plt.figure(2)
    mm = plt.plot(x,invest[:-1])
    plt.figure(3)
    aa = plt.plot(x,consumption[:-1])
    plt.show()
    
    return GDP, invest, consumption

# <codecell>

X0=sp.array([[0.,0.]])
shock=-.005
Y0=0
Xbar=sp.mat([[ks,ls]])
Zbar=0
alpha=.4
delta=.1
T=40
GDP_s, invest_s, consumption_s=impulseResponse(P,Q,R,S,NN,X0,Y0,shock,Xbar,Zbar,alpha,delta,T)

# <codecell>

P

# <codecell>

Q
    

# <codecell>

R

# <codecell>

S

# <codecell>


