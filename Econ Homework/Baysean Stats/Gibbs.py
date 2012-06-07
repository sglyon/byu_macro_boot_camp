"""
Created June 5, 2012

Author: Spencer Lyon
hi
"""
import numpy as np
import scipy as sp
import scipy.stats as st
import pylab

sp.set_printoptions(linewidth=140, suppress = True, precision = 5)

i = sp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = sp.array([5, 1, 5, 14, 3, 19, 1, 1, 4, 22])
t = sp.array([94.320, 15.720, 62.880, 125.76, 5.24,
			  31.44, 1.048, 1.048, 2.096, 10.48])
r = sp.array([.053, .064, .080, .111, .573, .604, .954, .954, 1.91, 2.099])


to_pass = np.vstack((y, t))

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
        obs: The data set over which you want to do the Gibbs sampling.

    Outputs:
    """
    alpha = 0.7
    d = 1.0
    c = 0.1

    k = obs.shape[1]

    beta = sp.rand(1) * 5
    theta = sp.rand(k) * .5

    beta_chain = np.empty((n,1))
    theta_chain = np.empty((n,k))

    for i in range(1,n):
        beta_chain[i] = st.invgamma.rvs(k * alpha + c,
                                scale = (1 / d + np.sum(theta_chain[i-1,:])))
        for j in range(k):
            theta_chain[i,j] = st.gamma.rvs(obs[0,j] + alpha,
                                        scale= 1 /(obs[1,j] + 1/beta_chain[i]))

    picky = np.arange(0,n-burn, 100)
    beta_chain = beta_chain[burn:]
    theta_chain = theta_chain[burn:, :]
    bet = beta_chain[picky]
    thet = theta_chain[picky,:]

    return [bet, thet]

def gen_plots(theta_ind):
    """
    This function generates a quick plot of the beta MC chain as well as the
    a histogram of its distribution. It does the same for the specified element
    of theta.

    Inputs:
        theta_ind: The theta (element of range 1 to 10) that you want to
                   visualize.

    Outputs:
        None: This funciton automaticlly generates and shows all plots.
    """

    beta, theta = gibbs(to_pass)

    pylab.hist(beta, bins = 40)
    pylab.show()
    pylab.plot(range(beta.size), beta)
    pylab.title('Beta chain')
    pylab.show()
    pylab.hist(theta[:, theta_ind], bins = 40)
    pylab.show()
    pylab.plot(range(theta[:, theta_ind].size), theta[:,theta_ind])
    pylab.title('Theta chain')
    pylab.show()

    return
