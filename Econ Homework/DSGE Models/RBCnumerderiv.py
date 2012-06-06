import scipy as sp
import numpy as np
from numpy.matlib import repmat
import math

def numericalDerivatives(func, ssVals, nx, ny, nz):
    epsilon = 1e-7
    ssVals   = np.mat(ssVals)
    T0 = max(abs(func(ssVals)))
    ssVal = ssVals


    length =  3*nx+2*(ny+nz)
    dev = repmat(ssVal.T,  1, length )

    for i in range(0,length):
        dev[i,i] += epsilon

    bigMat  = sp.zeros((2,8))
    bigMat[:,0] = ssVals[0,0] * (func(dev[:,0].T) - T0)/(1+T0)
    for i in range(1,length):
        if i<3*nx+2*ny+1:
            bigMat[:,i] = ssVals[0,i] * (func(dev[:,i].T) - T0)/(1+T0)
        else:
            bigMat[:,i] = (func(dev[:,i].T) - T0)/(1+T0)
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