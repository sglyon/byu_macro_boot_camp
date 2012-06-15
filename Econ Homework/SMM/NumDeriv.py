import scipy as sp
import numpy as np
from numpy.matlib import repmat
import math

beta  = 0.98
alpha = 0.4
delta = 0.10
zbar  = 0.
tao   = 0.05
expZbar = 1.
epsilon = .1
rho = 0.90

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

def func(vals,xsi,gamma,a):
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
	
	
	if a==0 and gamma==1:
		Euler1 = wt*(1-tao)/ct
	elif a==0:
		Euler1 = wt*(1-tao)/(ct**gamma)
	elif gamma==1 and xsi==1:
		Euler1 = ((1-lt)*wt*(1-tao))/(a*ct) - 1
	elif gamma==1:
		Euler1 = ((1-lt)**xsi * wt * (1-tao))/(ct * a) - 1
	elif xsi==1:
		Euler1 = ((1-lt)* wt * (1-tao))/(a*ct**gamma) -1
	else:
		Euler1 = wt*( 1-tao) * (1-lt)**xsi/(a*ct**gamma) - 1
		
	if gamma==1:
		Euler2 = beta*((ct/ct1)*(rt1-delta)*(1-tao)+1) - 1
	else:	
		Euler2 = beta*(((ct/ct1)**gamma)*((rt1 - delta)*(1-tao) + 1)) - 1
	
	return sp.array([Euler1, Euler2])

def numericalDerivatives(ssVals, nx, ny, nz,xsi,gamma,a):
    epsilon = 1e-7
    ssVals   = np.mat(ssVals)
    T0 = max(abs(func(ssVals,xsi,gamma,a)))
    ssVal = ssVals


    length =  3*nx+2*(ny+nz)
    dev = repmat(ssVal.T,  1, length )

    for i in range(0,length):
        dev[i,i] += epsilon

    bigMat  = sp.zeros((2,8))
    bigMat[:,0] = ssVals[0,0] * (func(dev[:,0].T,xsi,gamma,a) - T0)/(1+T0)
    for i in range(1,length):
        if i<3*nx+2*ny+1:
            bigMat[:,i] = ssVals[0,i] * (func(dev[:,i].T,xsi,gamma,a) - T0)/(1+T0)
        else:
            bigMat[:,i] = (func(dev[:,i].T,xsi,gamma,a) - T0)/(1+T0)
    bigMat /= epsilon


    AA = sp.mat(bigMat[0:ny,nx:2*nx])
    BB = sp.mat(bigMat[0:ny,2*nx:3*nx])
    CC = sp.mat(bigMat[0:ny,3*nx+ny:3*nx+2*ny])
    DD = sp.mat(bigMat[0:ny,3*nx*ny+nz+1:length])
    FF = sp.mat(bigMat[ny:ny+nx,0:nx])
    GG = sp.mat(bigMat[ny:ny+nx,nx:2*nx])
    HH = sp.mat(bigMat[ny:ny+nx,2*nx:3*nx])
    JJ = sp.mat(bigMat[ny:ny+nx,3*nx:3*nx+ny])
    KK = sp.mat(bigMat[ny:ny+nx,3*nx+ny:3*nx+2*ny])
    LL = sp.mat(bigMat[ny:ny+nx,3*nx+2*ny:3*nx+2*ny+nz])
    MM = sp.mat(bigMat[ny:ny+nx,3*nx+2*ny+nz:length])
    TT = sp.log(1+T0)

    if AA:
        AA = AA
    else:
        AA = sp.mat(sp.zeros((ny,nx)))

    if BB:
        BB = BB
    else:
        BB = sp.mat(sp.zeros((ny,nx)))

    if CC:
        CC = CC
    else:
        CC = sp.mat(sp.zeros((ny,ny)))

    if DD:
        DD = DD
    else:
        DD = sp.mat(sp.zeros((ny,nz)))



    return [AA,BB,CC,DD,FF,GG,HH,JJ,KK,LL,MM,TT]