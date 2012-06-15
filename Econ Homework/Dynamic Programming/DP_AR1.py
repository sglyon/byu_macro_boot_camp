import scipy as sp
import numpy as np
import scipy.special as sps
import tauchenhussey as th
from scipy import *
from scipy import linalg as la
from numpy import linspace
from scipy.stats import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#set up parameters, W matrix, etc. 
sigma2 = 0.25 
sigma  = sqrt(sigma2) 
mu     = 4*sigma 
M      = 7
rho    = 0.5 
basesigma = (0.5 +(rho/4))*sigma + (0.5 - (rho/4))*(sigma/((1-rho**2)**(0.5)))

n 		     = 100
w 		     = linspace(.01,1.,n)
eps, gamma   = th.tauchenhussey(M,mu,rho,sigma, basesigma)	
beta         = .9
VTplus1	     = zeros((n,M)) 

Warr	     = tile(vstack(w),(n,1,M))
Wprimearr    = tile(w,(M,n,1)).T
epsarr       = tile(eps.T,(n,n,1))
carr         = Warr - Wprimearr 
carrleq0     = carr<=0
carr[carrleq0.nonzero()] = 1e-10
uarr        = log(carr)


def dynamic_cinco(VT_old,beta,gamma,uarr,n):
	EVTplus1a	  = sp.dot(VT_old,gamma.T)
	EVTplusa          = EVTplus1a.reshape(n,1,M)
	EVTplus1arr	  = tile(EVTplusa,(1,n,1))
	EVTplus1arr[carrleq0] = -100
	epsuarr           = epsarr*uarr
	epsuarr[carrleq0] = -100
	
	VTarr       = epsuarr + beta*EVTplus1arr
	VT,psi      = VTarr.max(0), VTarr.argmax(0)
	return VT,psi

# VT,  psi, VTarr,  EVTplus1arr1 = dynamic_cinco(VTplus1,beta,gamma,uarr,n)
# VTm1,psi, VTarrm1,EVTplus1arr2 = dynamic_cinco(VT,beta,gamma,uarr,n)

def exercise5():
	count = 0
	err = 5
	tol = 1e-10
	VT_new = VTplus1
	while err>tol:
		VT_old = VT_new
		VT_new,psi = dynamic_cinco(VT_old,beta,gamma,uarr,n)	
		err     = (la.norm(VT_new-VT_old))**2;
		count +=1
		if count>2:
			break
	return VT_new, psi

VT_new, psi = exercise5()

X, Y = np.meshgrid(eps, w)
fig = plt.figure()
ax = fig.add_subplot(111, projection ='3d')

surface = ax.plot_surface(X, Y, psi, rstride = 1, cstride = 1)
plt.title('Policy Function')
ax.set_xlabel('Taste Shock')
ax.set_ylabel('Cake Today')
plt.show()
