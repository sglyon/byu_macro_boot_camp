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

w = np.array([1.6907, 1.7242, 1.7552, 1.7842, 1.8113, 1.8369, 1.8610, 1.8839])
y = np.array([6, 13, 18, 28, 52, 53, 61, 60.0], dtype = float)
n = np.array([59, 60, 62, 56, 63, 59, 62, 60.0], dtype = float)

a0 = .25
b0= 4.
c0 = 2.
d0 = 10.
e0 = 2.000004
f0 = 1000

sig = np.diag([0.00012, 0.033, 0.10])

theta_1 = np.array([1.8, -3.9, -1], dtype = float)
theta_2 = np.array([1.5, -4, -2.8])
theta_3 = np.array([2.1, -1.4, 1.5])

def MH_Gib(n=10000, burn=1000):
    """
    This function estimates the distributions for the parameters in the flour
    beetle mortality study using a Gibbs sampler with Metropolis-Hastings
    algorithms used to sample from the posterior distributions for each
    parameter.

    Inputs:
        n: The number of elements to be generated in the Gibbs sampler.
        burn: The length of the burn in for both the MH and Gibbs routines.

    Outputs:
        theta:
    """
    
