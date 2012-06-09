import scipy as sp
from scipy import stats as st

def discretenorm(N,mu,sig):
	"""
	Function discretenorm

	Purpose:    Generates a discretized support of a normally distributed
				i.i.d. process with N evenly spaced nodes
				eps(t) ~ N(mu,sigma2)
				where the vector eps is the support and Gamma is the pdf
				of the support. The pdf is generated as the evenly spaced
				bins around each of the nodes in eps.

	Format:     [eps, Gamma] = discretenorm(N,mu,sigma)

	Input:      N         scalar, number of nodes for eps
				mu        scalar, unconditional mean of process
				sigma     scalar, std. dev. of epsilons

	Output:     eps       1 x N vector, nodes for epsilon
				Gamma     1 x N vector, discrete pdf of eps
						  Gamma(eps) = Pr(eps=eps_n)

	Author:     Benjamin J. Tengelsen, Brigham Young University
				Richard W. Evans, Brigham Young University
				October 2009 (updated October 2009)
	"""

	# determine minimum and maximum points
	epsmax   = mu + 3*sig
	epsmin   = mu - 3*sig

	# determine the step size between points
	epsinc	 = (epsmax - epsmin)/float(N-1)
	# find the points
	eps      = sp.linspace(epsmin,epsmax,N)
	# find the discrete `bin' around each point
	epsbins  = eps[1:] - epsinc/2.

	# evaluate volume of each bin
	Gammacdf = st.norm.cdf(epsbins,mu,sig)
	Gamma    = sp.zeros(N)
	Gamma[0] = Gammacdf[0]
	Gamma[6] = 1 - Gammacdf[5]
	for i in range(1,N-1):
		Gamma[i] = Gammacdf[i] - Gammacdf[i-1]
	return eps, Gamma
