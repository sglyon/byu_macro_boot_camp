import scipy as sp
import math
from random import gauss
import matplotlib.pyplot as plt
import numpy as np
from scipy import mean

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


def MonteCarlo(P,Q,R,S,N,X0,Z,Xbar,alpha,delta,reps,T):
	C = sp.zeros((T,reps))
	L = sp.zeros((T,reps))
	Lauto = []
	Cauto = []
	Wauto = []
	LWcorr = []


	for i in range(reps):
		if i % 500 ==0:
			print 'Simulation #: ', i
		X=[]
		X = Xgen(X0,Z[:,i],P,Q,Xbar)

		#find C,L for each time period
		consumption,labor,wage=CLgen(X,Z[:,i],alpha,delta)


		#Consumption autocorrelation coefficients for each series
		#appended to the empty Cauto list
		#cauto = sp.corrcoef(consumption[0:-1],consumption[1:])
		cauto = sp.mean(consumption)
		Cauto.append(cauto)

		#Labor autocorrelation coefficients for each series
		#appended to the empty Lauto list
		#lauto = sp.corrcoef(labor[0:-1],labor[1:])
		lauto = sp.mean(labor)
		Lauto.append(lauto)

		#Wage autocorrelation for each series
		#appended to the empty Lauto list
		#wauto = sp.corrcoef(wage[0:-1],wage[1:])[0,1]
		wauto = sp.mean(wage)
		Wauto.append(wauto)
		
		#Wage and Labor correlation coefficient for each series
		#appended to the empty LWcorr list
		lwcorr = sp.corrcoef(labor,wage)
		LWcorr.append(lwcorr)

	#What we will return, arrays of the auto and correlation coefficients
	CAUTO = sp.array(Cauto)

	LAUTO = sp.array(Lauto)

	WAUTO = sp.array(Wauto)
	
	LWCORR = sp.array(LWcorr)

	return CAUTO,LAUTO,WAUTO,LWCORR

def CLgen(X,Z,alpha, delta):
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
	W = sp.zeros(t)
	#solve for Y in each period t
	for i in range(t):
		Y[i] = (K[i]**alpha)*((sp.exp(Z[i,0])*L[i])**(1.-alpha))
	#solve for I in each period t
	for i in range(t-1):
		I[i] = K[i+1] - (1. - delta)*K[i]
	#solve for C in each period t
	for i in range(t-1):
		C[i] = Y[i] - I[i]
	for i in range(t):
		W[i] = (1.-alpha)*((K[i]/L[i])**alpha)*((sp.exp(Z[i,0]))**(1.-alpha))
	return C,L,W


