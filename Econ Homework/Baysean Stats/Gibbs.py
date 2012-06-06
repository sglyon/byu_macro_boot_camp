"""
Created June 5, 2012

Author: Spencer Lyon
"""
import numpy as np
import scipy as sp
import scipy.stats as st

i = sp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = sp.array([5, 1, 5, 14, 3, 19, 1, 1, 4, 22])
t = sp.array([94.320, 15.720, 62.880, 125.76, 5.24,
			  31.44, 1.048, 1.048, 2.096, 10.48])
r = sp.array([.053, .064, .080, .111, .573, .604, .954, .954, 1.91, 2.099])
theta = y*t

def gibbs(obs, n=10000, burn=1000):
    """
    This function applies the Gibb's Sampling algorithm to the data passed in.
    By default we will take 20000 monte carlo samples, drop the first 1000 (for
    the burn in) and then take every 200th entry after that. That leaves us with
    a sample with 95 entries.

    We will assume that for this model that we have the following distributions:
        liklihood: f(y|theta) ~P(theta)
        Prior: g(theta|beta) ~ G(alpha, beta)
        Hyperprior: h(beta) ~ invGamma (c, d)

    Inputs:
        n: The number of elements to be generate in the MC process
        burn: The number of elements ot be thrown out to account for burn-in
        obs: The observation that is to be used in generating the sample.

    Outputs:
    """
    # Define parameters
    alpha = 0.7
    d = 1.0
    c = 0.1

    beta = np.empty(n)
    for i in range(n):
        beta[i] = st.invgamma.rvs(alpha + c, 1/(obs + 1/d))
