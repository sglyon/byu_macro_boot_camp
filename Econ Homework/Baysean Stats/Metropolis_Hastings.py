"""
Created June 5, 2012

Author: Spencer Lyon
"""
import numpy as np
import scipy as sp
import scipy.stats as st

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


def compute_r(theta):
    """
    This function implements the posterior distribution as derived at the
    beginning of the problem

    Inputs:
        theta: The variable of parameters to be used in evaluating the function.
           It will be in this form:
               theta = [mu, 1/2* log(sigma_squared), log(m1)]


    Outputs:
    r: The coefficient r computed using the equations in the problem.
    """

    theta_star = np.random.multivariate_normal(theta, sig)

    # Taking the log before I calculate h(theta) [numerical precision errors].
    r_num = np.sum( y * np.log(g(w,theta_star)) + (n-y) * np.log((1 - g(w, theta_star)))) + \
            a0 * theta_star[2] - 2  * e0 * theta_star[1] + \
            -1/2 * ((theta_star[0] - c0)/d0)**2 - np.exp(theta_star[2])/b0 \
                    - np.exp( -2 * theta_star[1]) / f0

    r_denom = np.sum( y * np.log(g(w,theta)) + (n-y) * np.log((1 - g(w, theta)))) + \
            a0 * theta[2] - 2  * e0 * theta[1] + \
            -1/2 * ((theta[0] - c0)/d0)**2 - np.exp(theta[2])/b0 \
                    - np.exp( -2 * theta[1]) / f0

    #print np.exp(r_num)
    #print np.exp(r_denom)

    #r_num = np.product( g(w,theta_star)**y * (1 - g(w, theta_star))**(n - y))* \
    #        np.exp(a0 * theta_star[2] - 2  * e0 * theta_star[1]) * \
    #        np.exp(-1/2 * ((theta_star[0] - c0)/d0)**2 - np.exp(theta_star[2]/b0)
    #                - np.exp( -2 * theta_star[1]) / f0)
    #
    #r_denom = np.product( g(w,theta)**y * (1 - g(w,theta))**(n - y)) * \
    #        np.exp(a0 * theta[2] - 2 * e0 * theta[1]) * \
    #        np.exp(-1/2 * ((theta[0] - c0) / d0) **2 - np.exp(theta[2] / b0)
    #                - np.exp( -2 * theta[1]) / f0)

    r = np.exp(r_num - r_denom)

    return r, theta_star

def main(theta, n = 10000):
    """

    """
    theta_chain = np.empty((3,n))
    theta_old = theta
    theta_chain[:,0] = theta_old
    accept = 0
    for i in range(n):
        r, theta_star  = compute_r(theta_old)
        test = sp.rand(1)
        if r > test:
            theta_old = theta_star
            accept += 1
        else:
            theta_old = theta_old
        theta_chain[:,i] = theta_old
    print 'acceptance rate = ', float(accept)/n

    return theta_chain
