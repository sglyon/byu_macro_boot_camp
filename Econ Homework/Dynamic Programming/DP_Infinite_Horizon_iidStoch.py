"""
Created June 8, 2012

Author: Spencer Lyon
"""
import numpy as np
import discretenorm as dn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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

## --------------------------------Problem 1--------------------------------- ##
sigma = np.sqrt(0.25)
mu = 4 * sigma
nodes = 7
beta = 0.9
tol = 10e-9

eps, gamma = dn.discretenorm(nodes, mu, sigma)

## --------------------------------Problem 2--------------------------------- ##
w = np.array(np.arange(0.01, 1.01, .01))
N = w.size
vtp1 = np.zeros((N,nodes))

# Create probability matrix for epsilon'
E_mat = np.tile(gamma, (N, 1))

# Integrate out epsilon' and get just the E[v(w')] vector.
Evtp1_prime = np.sum(vtp1 * E_mat, axis = 1)

# Flip it so the w' is the third dimension. Then repeat it in w, epsilon dims.
Evtp1_prime = np.reshape(Evtp1_prime, (1,1,N))
Evtp1_mat = np.tile(Evtp1_prime, (N, nodes, 1))

# Create Utility matrix.
# This is three dimensional. 1: w, 2: e, 3: w'
c_mat = np.empty((N, nodes, N))
for i in range(N):
    for j in range(nodes):
        for k in range(N):
            if w[i] - w[k] > 0:
                c_mat[i,j,k] = eps[j] * (w[i] - w[k])
            else:
                c_mat[i, j, k] = 10e-10

util_mat = np.log(c_mat)

vt, polt = my_max(util_mat + beta * Evtp1_mat, axis = 2)

## --------------------------------Problem 3--------------------------------- ##
def my_norm(v_old, v_new):
    """
    This function evaluates the norm delta = ||vt - vt1|| that measures the
    distance between two value functions. We will make this definition as
    the sum of squared distances of the flattened Value funciton matricies.

    Inputs:
        v_old: The first of two value functions for which you want the norm
        v_new: The second of two value functions for which you want the norm.

    Outputs:
        delta: The norm between the two value functions.
    """
    flat = (v_old - v_new).flatten()

    return np.dot(flat, flat)

deltat = my_norm(vtp1, vt)

## --------------------------------Problem 4--------------------------------- ##
Evt_prime = np.sum(vt * E_mat, axis = 1)
Evt_prime = np.reshape(Evt_prime, (1,1,N))
Evt_mat = np.tile(Evt_prime, (N, nodes, 1))
vt_1, polt_1 = my_max(util_mat + beta * Evt_mat, axis = 2)
deltat_1 = my_norm(vt, vt_1)

## --------------------------------Problem 5--------------------------------- ##
Evt_1_prime = np.sum(vt_1 * E_mat, axis = 1)
Evt_1_prime = np.reshape(Evt_1_prime, (1,1,N))
Evt_1_mat = np.tile(Evt_1_prime, (N, nodes, 1))
vt_2, polt_2 = my_max(util_mat + beta * Evt_1_mat, axis = 2)
deltat_2 = my_norm(vt_1, vt_2)
print deltat_1 - deltat_2

## --------------------------------Problem 6--------------------------------- ##
vnew, pnew = vt_2, polt_2
theNorm = deltat_2
iterations = 2
while theNorm > tol:
    vold, pold = vnew, pnew
    Ev_prime_new = np.sum(vold * E_mat, axis = 1)
    Ev_prime_new = np.reshape(Ev_prime_new, (1,1,N))
    Ev_mat_new = np.tile(Ev_prime_new, (N, nodes, 1))
    vnew, pnew = my_max(util_mat + beta * Ev_mat_new, axis = 2)
    theNorm = my_norm(vnew, vold)
    iterations += 1
    print iterations


## --------------------------------Problem 7--------------------------------- ##
# TODO. Make the plot look pretty.
X, Y = np.meshgrid(eps[1:], w[1:])
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

surface = ax.plot_surface(X, Y, pnew[1:,1:], rstride = 1, cstride = 1)
plt.title('Policy Function')
ax.set_xlabel('Taste Shock')
ax.set_ylabel('Cake Today')
plt.show()
