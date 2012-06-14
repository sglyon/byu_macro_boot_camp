"""
Created June 14, 2012

Author: Spencer Lyon
"""
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from numpy import sqrt, pi, exp

## Define Parameters and functions
mu, tao, y,  sigma, n1, n2= 2., 1., 6., 1., 1., 10.

def normpdf(x, mu, sigma):
    u = (x-mu)/abs(sigma)
    y = (1/(sqrt(2*pi)*abs(sigma)))*exp(-u*u/2)
    return y

x = np.linspace(0, 50.0 ,500)

theta = normpdf(x, mu, sigma)

post_mean_1 = (sigma**2 * mu + n1 * tao**2 * y)/(sigma**2 + n1 * tao**2)
post_variance_1 = (sigma**2 * tao**2)/(sigma**2 + n1 * tao**2)

post_mean_10 = (sigma**2 * mu + n2 * tao**2 * y)/(sigma**2 + n2 * tao**2)
post_variance_10 = (sigma**2 * tao**2)/(sigma**2 + n2 * tao**2)

## n = 1 case
plt.plot(x, normpdf(x, mu, tao**2), label='Prior')
plt.plot(x, normpdf(x, theta, sigma**2), label = 'Liklihood')
plt.plot(x, normpdf(x, post_mean_1, post_variance_1), label = 'Posterior')
plt.legend(loc=0)
plt.title('When n=1')
plt.show()


## n = 10 case
plt.figure()
plt.plot(x, normpdf(x, mu, tao**2), label='Prior')
plt.plot(x, normpdf(x, theta, sigma**2), label = 'Liklihood')
plt.plot(x, normpdf(x, post_mean_10, post_variance_10), label = 'Posterior')
plt.legend(loc=0)
plt.title('When n=10')
plt.show()
