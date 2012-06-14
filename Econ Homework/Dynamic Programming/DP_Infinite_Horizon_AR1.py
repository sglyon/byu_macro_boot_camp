"""
Created June 11, 2012

Author: Spencer Lyon
"""
from __future__ import division
import numpy as np
import scipy as sp
import tauchenhussey as th
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
sp.set_printoptions(linewidth=140, suppress = True, precision = 5)

# Parameters
beta = 0.9
tol = 10e-9

# Useful funcitons
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

## --------------------------------Problem 1--------------------------------- ##
w = np.array(np.arange(0.01, 1.01, .01))
N = w.size
nodes = 7
sigma = 1/2.
mu = 4 * sigma
rho = 1/2.
base_sigma = (0.5 + rho/4.0) * sigma + (0.5 - rho/4.0) * (sigma/(np.sqrt(1 - rho**2)))
support, p_mat = th.tauchenhussey(nodes, mu, rho, sigma, base_sigma)
p_cube = np.reshape(p_mat, (nodes, nodes, 1))
p_cube = np.tile(p_cube, (1, 1, N))

## --------------------------------Problem 2--------------------------------- ##


# Create Utility matrix.
# This is three dimensional. 1: w, 2: e, 3: w'
c_mat = np.empty((N, nodes, N))
for i in range(N):
    for j in range(nodes):
        for k in range(N):
            if w[i] - w[k] >= 0:
                c_mat[i,j,k] = support[j] * (w[i] - w[k])
            else:
                c_mat[i, j, k] = 10e-10

util_mat = np.log(c_mat)

changes = 0
for j in range(util_mat.shape[1]):
    for i in range(util_mat.shape[0]):
        for k in range(util_mat.shape[2]):
            if k >= i:
                util_mat[i,j,k] = -10e20
                changes +=1
print changes

# Create value function matrix. v(w', e')
vtp1 = np.zeros((N,nodes))

# This funciton will do a contraction.
def do_iteration(v):
    """
    This funciton takes in a value function (v) and does the contraction to
    produce a new value function v'(w',e')

    Inputs:
        v: The old value funciton. This is an N x nodes vector where N is the
           dimension of the state space and nodes is the number of values in the
           support of epsilon.

    Outpus:
        v_prime: The new value function
        pol_prime: The new policy function.
    """
    v_temp = np.reshape(v,(1,nodes, N))
    v_prime= np.tile(v_temp, (nodes, 1,1))
    v_mat = np.sum(p_cube * v_prime, axis = 1)
    v_mat = np.reshape(v_mat, (1, nodes, N))
    Ev_mat = np.tile(v_mat, (N, 1,1))


    v_prime, pol_prime = my_max(util_mat + beta * Ev_mat, axis = 2)
    return v_prime, pol_prime

# Call the function above to get the contracted values.
vt, polt = do_iteration(vtp1)


## --------------------------------Problem 3--------------------------------- ##

# See my_norm function from above. This is the norm asked for.
deltat = my_norm(vtp1, vt)

## --------------------------------Problem 4--------------------------------- ##
vt_1, polt_1 = do_iteration(vt)
deltat_1 = my_norm(vt, vt_1)

print 'deltaT = ', deltat, 'deltaT_1 = ', deltat_1

## --------------------------------Problem 5--------------------------------- ##
vt_2, polt_2 = do_iteration(vt_1)
deltat_2 = my_norm(vt_1, vt_2)

print 'deltaT = ', deltat, 'deltaT_1 = ', deltat_1, 'deltaT_2 = ', deltat_2,


## --------------------------------Problem 6--------------------------------- ##
vnew, pnew = vt_2, polt_2
theNorm = deltat_2
iterations = 2
while theNorm > tol:
    vold, pold = vnew, pnew
    vnew, pnew = do_iteration(vold)
    theNorm = my_norm(vnew, vold)
    iterations += 1
<<<<<<< Updated upstream
print '\nDone!\n After %.2f iterations the current value of delta = %f'\
                                                % (iterations, theNorm)

## --------------------------------Problem 7--------------------------------- ##
X, Y = np.meshgrid(support, w)
fig = plt.figure()
ax = fig.add_subplot(111, projection ='3d')

surface = ax.plot_surface(X, Y, pnew, rstride = 1, cstride = 1)
=======
print '\nDone!\nCurrent value of delta after %.2f iterations is = %f' %\
       (iterations, theNorm)

## --------------------------------Problem 7--------------------------------- ##
X, Y = np.meshgrid(support[2:], w[2:])
fig = plt.figure()
ax = fig.add_subplot(111, projection ='3d')

surface = ax.plot_surface(X, Y, pnew[2:,2:], rstride = 1, cstride = 1)
>>>>>>> Stashed changes
plt.title('Policy Function')
ax.set_xlabel('Taste Shock')
ax.set_ylabel('Cake Today')
plt.show()
