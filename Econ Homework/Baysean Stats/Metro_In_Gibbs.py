"""
Created June 7, 2012

Author: Spencer Lyon, Chase Coleman
"""
from __future__ import division
import numpy as np
import scipy as sp
import scipy.stats as st
import pylab as pl
import Gibbs as gib
import Metropolis as met
from copy import copy

## Define constants, givens, and paramters.
w = np.array([1.6907, 1.7242, 1.7552, 1.7842, 1.8113, 1.8369, 1.8610, 1.8839])
y = np.array([6, 13, 18, 28, 52, 53, 61, 60.0], dtype = float)
n = np.array([59, 60, 62, 56, 63, 59, 62, 60.0], dtype = float)



a0 = .25
b0= 4.
c0 = 2.
d0 = 10.
e0 = 2.000004
f0 = 1000

sigs = np.array([0.00012, 0.033, 0.10])

theta_1 = np.array([1.8, -3.9, -1], dtype = float)
theta_2 = np.array([1.5, -4, -2.8])
theta_3 = np.array([2.1, -1.4, 1.5])

## Define necessary things for the MH algorithm.

# Define Liklihood function.
def g(w, theta):
    """
    This is a simple implementation of the logit function given us in the
    problem. The functional form is:
        g(w) = (exp(x) / (1 + exp(x)))**m1
        where x = (w - mu) / sigma

    Inputs:
        w: The w at which you would like to evaluate the function

    Outputs:
        val: The value of the function at the given w.
    """
    mu = theta[0]
    sigma = np.exp(theta[1])
    m1 = np.exp(theta[2])
    x = (w - mu) / sigma

    return (np.exp(x) / (1 + np.exp(x))) ** m1

# Define univariate metropolis 'r' coefficients.
def new_th(theta, element):
    """
    This function computes a new th1 using a univariate umping distribution.

    Inputs:
        theta: The current value of theta. This is a 3 element NumPy array.
        element: The element you need to replace with an updated value. This
                 will be calculated with a univariate normal distribution with
                 mean theta[element] and variance sigs[element]
    Returns:
        new_theta: The is the updated array with the new value replaced.
    """
    # Copy incoming theta.
    theta_new = copy(theta)

    # Change one element in theta, by drawing from normal distribution.
    to_change = float(theta[element])
    changed = np.random.normal(to_change, sigs[element])

    # Change element in theta_new and return.
    theta_new[element] = changed

    return theta_new

def compute_r(theta, element):
    """
    This function computes the marginal posterior distribution for theta_1 in
    the flour beetle model.

    Inputs:
        theta: This is the original theta for which r will be computed.
        element: The element of theta that will be varied in the univariate MH
                 algorithm.

    Returns:

    """
    theta_star = new_th(theta, element)
    new_th_el = theta_star[element]

    # Taking the log before I calculate h(theta) [numerical precision errors].
    r_num = np.sum( y * np.log(g(w,theta_star)) + (n-y) * np.log((1 - g(w, theta_star)))) + \
            a0 * theta_star[2] - 2  * e0 * theta_star[1] + \
            -1/2 * ((theta_star[0] - c0)/d0)**2 - np.exp(theta_star[2])/b0 \
                    - np.exp( -2 * theta_star[1]) / f0

    r_denom = np.sum( y * np.log(g(w,theta)) + (n-y) * np.log((1 - g(w, theta)))) + \
            a0 * theta[2] - 2  * e0 * theta[1] + \
            -1/2 * ((theta[0] - c0)/d0)**2 - np.exp(theta[2])/b0 \
                    - np.exp( -2 * theta[1]) / f0

    r = np.exp(r_num - r_denom)

    return r, new_th_el

def mini_MH(theta, element, n, burn):
    """

    """
    theta_old = theta
    mini_th_chain = np.empty(n)
    mini_th_chain[0] = theta_old[element]

    for i in range(n):
        r, theta_star = compute_r(theta_old, element)
        test = sp.rand(1)
        if r > test:
            theta_old[element] = theta_star
        else:
            theta_old[element] = theta_old[element]
        mini_th_chain[i] = theta_old[element]

    taking = np.random.randint(n-burn, n)

    new_th_sample = mini_th_chain[taking]

    return new_th_sample


def MH_Gib(theta_init, N=10000, n=2000, burn=1000):
    """
    This function estimates the distributions for the parameters in the flour
    beetle mortality study using a Gibbs sampler with Metropolis-Hastings
    algorithms used to sample from the posterior distributions for each
    parameter.

    Inputs:
        theta_init: The theta that will kick off the whole process.
        **N: The number of elements to be generated in the Gibbs sampler and
           ultimately returned in the final chain.
        **n: The number of elements to be used when calculating the draw from MH
        **burn: The length of the burn in for both the MH and Gibbs routines.

    Outputs:
        theta_chain: The final chain of length N generated in the sampler.

    Notes:
        ** means optional parameter.
    """

    theta_chain = np.empty((3, n))
    theta_old = theta_init
    theta_chain[:,0] = theta_old

    for i in range(1, N):
        theta_chain[0,i] = mini_MH(theta_chain[:,i-1], 0, n, burn)

        theta_chain[1,i] = mini_MH([theta_chain[0, i],
                                    theta_chain[1, i-1],
                                    theta_chain[2, i-1]], 1, n, burn)

        theta_chain[2,i] = mini_MH([theta_chain[0, i],
                                    theta_chain[1, i],
                                    theta_chain[2, i-1]], 1, n, burn)
        if i % 500 ==0:
            print 'Iteration', i

    return theta_chain

MH_Gib(theta_2)



