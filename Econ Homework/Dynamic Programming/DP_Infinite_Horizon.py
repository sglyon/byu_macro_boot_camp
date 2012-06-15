"""
Created June 7, 2012

Author: Spencer Lyon
"""
import numpy as np
from numpy.matlib import repmat
import matplotlib.pyplot as plt

# Define parameters
beta = 0.9
tol = 1.0e-9


w = np.array(np.arange(0.01, 1.01, .01), ndmin = 2)
N = w.size


## Problem 2
v_t1 = np.zeros((N,1))
temp_mat = repmat(w, N, 1)
util_mat = np.log(temp_mat.T - temp_mat)

for i in range(N):
    for j in range(N):
        if j >= i:
            util_mat[i,j] = -10e10

def my_max(array, axis=1):
    """
    This function implements the equilvalent of Matlab's max function.

    Inputs:
        array: This can be a N-dimensional NumPy array for which you want to
               find the max along a certain dimension.
        axis: The axis along which to find the max value and index. Default
              value is 1.

    Outputs:
        max_vals: The maximum values found along the dimension.
        arg_maxes: The locaiton of the max_vals along a dimension.
    """
    return np.array([np.max(array, axis=axis), np.argmax(array, axis=axis)])

vt, polt = my_max(util_mat + beta * repmat(v_t1.T,N, 1))

## problem 3
def norm(vt, vt1):
    """
    This function evaluates the norm delta = ||vt - vt1|| that measure the
    distance between two value functions. We will make this definition as
    the sum of squared distances.

    Inputs:
        vt: The first of two value functions for which you want the norm
        vt1: The second of two valuefunctions for which you want the norm.

    Outputs:
        delta: The norm between the two value functions.
    """
    vt = np.reshape(vt, (N,1))
    vt1 = np.reshape(vt1, (N,1))
    return np.dot( (vt.T - vt1.T), (vt - vt1))

deltat = norm(vt, v_t1)

## Problem 4
vt_1, polt_1 = my_max(util_mat + beta * repmat(np.reshape(vt, (1,N)), N, 1))
deltat_1 = norm(vt_1, vt)

differencet_1    = deltat - deltat_1

## Problem 5
vt_2, polt_2 = my_max(util_mat + beta * repmat(np.reshape(vt_1, (1,N)), N, 1))
deltat_2 = norm(vt_2, vt_1)

differencet_2 = deltat_1 - deltat_2

## Problem 6
vnew, pnew = vt_2, polt_2
theNorm = deltat_2
while theNorm > tol:
    vold, pold = vnew, pnew
    vnew, pnew = my_max(util_mat + beta * repmat(np.reshape(vold, (1,N)), N, 1))
    theNorm = norm(vnew, vold)

## Problem 7
# TODO: Do this again. Follow instructions this time.
plt.plot(np.arange(pnew.size)[1:], pnew[1:])
plt.title('Policy function')
plt.show()

plt.plot(np.arange(vnew.size)[1:], vnew[1:])
plt.title('Value function')
plt.show()
