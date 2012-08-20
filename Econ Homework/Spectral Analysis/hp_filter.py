"""
Created May 30, 2012

Author: Spencer Lyon
"""
import numpy as np
import scipy as sp
import scipy.sparse.linalg as spla


def hp_filter(data, lamb = 1600):
    """
    This function will apply a Hodrick-Prescott filter to a dataset.
    The return value is the filtered data-set found according to:
        min sum((X[t] - T[t])**2 + lamb*((T[t+1] - T[t]) - (T[t] - T[t-1]))**2)
          T   t

    T = Lambda**-1 * Y

    Inputs:
        data: The data set for which you want to apply the HP_filter. This must
              be a numpy array.
        lamd: This is the value for lambda as used in the equation.

    Outputs:
        T: The solution to the minimization equation above (the trend).
        Cycle: This is the 'stationary data' found by Y - T.

    Notes:
        This function implements sparse methods to be efficient enough to handle
        very large data sets.
    """
    Y = np.array(data)
    if Y.ndim >1:
        Y = Y.squeeze()
    lil_t = len(Y)
    big_Lambda = sp.sparse.eye(lil_t, lil_t)
    big_Lambda = sp.sparse.lil_matrix(big_Lambda)

    # Use FOC's to build rows by group. The first and last rows are similar.
    # As are the second-second to last. Then all the ones in the middle...
    first_last = np.array([1+ lamb, -2*lamb, lamb])
    second = np.array([-2* lamb, (1+ 5*lamb), -4*lamb, lamb])
    middle_stuff = np.array([lamb, -4.*lamb, 1+6*lamb, -4*lamb, lamb])

    #--------------------------- Putting it together --------------------------#

    # First two rows
    big_Lambda[0,0:3] = first_last
    big_Lambda[1,0:4] = second

    # Last two rows. Second to last first
    big_Lambda[lil_t-2, -4:] = second
    big_Lambda[lil_t-1, -3:] = first_last

    # Middle rows
    for i in range(2, lil_t-2):
        big_Lambda[i, i-2:i+3] = middle_stuff

    # spla.spsolve requires csr or csc matrix. I choose csr for fun.
    big_Lambda = sp.sparse.csr_matrix(big_Lambda)

    T = spla.spsolve(big_Lambda, Y)

    Cycle = Y - T

    return T, Cycle
