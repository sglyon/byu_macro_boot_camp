import scipy as sp
import sympy
from sympy import Symbol, nsolve
import sympy.mpmath as mp

def solveSS(gamma,xsi,a):
	beta  = 0.98
	alpha = 0.4
	delta = 0.10
	zbar  = 0.
	tao   = 0.05
	expZbar = 1.
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
	SOLSS = nsolve((f1,f2,f3,f4,f5,f6),(Css, rss, lss, wss, Tss, kss),
                   (.75,.12,.55,1.32,.041,4.05))

	cs = float(SOLSS[0])
	rs = float(SOLSS[1])
	ls = float(SOLSS[2])
	ws = float(SOLSS[3])
	Ts = float(SOLSS[4])
	ks = float(SOLSS[5])
	return cs,rs,ls,ws,Ts,ks
