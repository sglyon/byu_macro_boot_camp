"""
Created June 5, 2012

Author: Spencer Lyon
"""

import numpy as np
import scipy as sp
import scipy.stats as st

w = np.array([1.6907, 1.7242, 1.7552, 1.7842, 1.8113, 1.8369, 1.8610, 1.8839])
y = np.array([6, 13, 18, 28, 52, 53, 61, 60.0], dtype = float)
n = np.array([59, 60, 62, 56, 63, 59, 63, 60.0], dtype = float)

a0 = .25
b0= 4.
c0 = 2.
d0 = 10.
e0 = 2.000004
f0 = 1000

sig = np.diag([0.00012, 0.33, 0.10])

theta_1 = np.array([1.8, -2.7, -1])
theta_2 = np.array([1.5, -4, -2.8])
theta_3 = np.array([2.1, -1.4, 1.5])

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
    sigma = np.sqrt(2 * np.exp(theta[1]) )
    m1 = np.exp(theta[1])
    x = (w - mu) / sigma

    return (np.exp(x) / (1 + np.exp(x))) ** m1

# Some changes

def compute_r(theta):
    """
    This function implements the posterior distribution as derived at the
    beginning of the problem

    Inputs:
        theta: The variable of parameters to be used in evaluating the function.
           It will be in this form:
               theta = [mu, 1/2* log(sigma_squared), log(m1)]


    Outputs:
    r: The coefficient r computed at using the equations in the problem.
    """

    theta_star = np.random.multivariate_normal(theta, sig)

    r_num = np.product( g(w,theta_star)**y * (1 - g(w, theta_star))**(n - y))* \
            np.exp(a0 * theta_star[2] - 2  * e0 * theta_star[1]) * \
            np.exp(-1/2 * ((theta_star[0] - c0)/d0)**2 - np.exp(theta_star[2]/b0)
                    - np.exp( -2 * theta_star[1]) / f0)

    r_denom = np.product( g(w,theta)**y * (1 - g(w,theta))**(n - y)) * \
            np.exp(a0 * theta[2] - 2 * e0 * theta[1]) * \
            np.exp(-1/2 * ((theta[0] - c0) / d0) **2 - np.exp(theta[2] / b0)
                    - np.exp( -2 * theta[1]) / f0)

    r = np.exp(np.log(r_num) - np.log(r_denom))

    return r

    #np.product( g(w)**y * ( )
