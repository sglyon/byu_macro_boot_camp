"""
Created June 14, 2012

Author: Spencer Lyon
"""
import numpy as np
import scipy.optimize as opt
import Bandpass_Filter as bpf
import hp_filter as hp
import DSGE
import MonteCarloSimulation as mcs
import NumDeriv as nd
import SolveSS as sols
import UhligSolve as lig
import pandas as pan
from pandas.io.data import DataReader as dr
import datetime

##-------------------- Step 1. Get data and filter it.------------------------##
# We will use Wage data: ECIWAG
# Also business labor: PRS84006173
# and Consumption of fixed capital: COFC

start_date = datetime.datetime(2000, 1, 1)
wage_Data= np.asarray(dr('ECIWAG', 'fred', start = start_date)['VALUE'])
labor_Data = np.asarray(dr('PRS84006173', 'fred', start = start_date)['VALUE'])
consump_Data = np.asarray(dr('COFC', 'fred', start = start_date)['VALUE'])

filt_wage = hp.hp_filter(wage_Data)[1]
filt_labor = hp.hp_filter(labor_Data)[1]
filt_consump = hp.hp_filter(consump_Data)[1]

moments = np.array([np.mean(filt_consump), np.mean(filt_labor),
                    np.mean(filt_wage)   ,np.var(filt_consump)])

##------------------------Step 2. Find steady state --------------------------##
def get_current_ss(beta):
    """
    This calls SolveSS.py to get the steady state for the particular parameter
    vector beta.

    Inputs:
        beta: This is a 3 element array with the value for gamma in the first
              place, xsi in the second, and a in the third.
    """
    return sols.solveSS(beta[0], beta[1], beta[2])


##---------------Step 3. Use law of motion to get Z history-------------------##
rho = .9
sigma_sq = 0.004
sigma = np.sqrt(sigma_sq)
mu = 0.0
z_bar = 0.0

def new_z_shock(init, t):
    """
    This applies the law of motion to generate a chain of shocks for t periods.
    The shocks are distributed iid N(mu, sigma)

    Inputs:
        init: the initial shock value. Where you want to start the process.
        t: How long the shock chain needs to be

    Returns:
        Z: THe chain of Z's that was asked for.
    """
    Z = np.empty(2* t)
    Z[0] = 0
    Z[1] = init
    for i in range(2, 2*t):
        Z[i] = rho * Z[i-1] + np.random.normal(mu, sigma)
        for j in range(t):
            Z[j] = Z[i] + z_bar + (1-rho)

    return Z


# Now we need to use this function to create a with M chains each T long....
def z_matrix(init, t, n=10000):
    """
    This function uses the law of motion (defined above) to generate n chains
    each of length t.

    Inputs:
        init: (scalar) where to start the process form on each iteration
        t: (int) How long each chain needs to be
        n: (int) The number of chains that you want in the end.

    returns:
        zmat: A NumPy array that contains the desired information
    """

    zmat = []
    count = 0
    for i in range(n):
        z = new_z_shock(init, t)
        z = np.vstack(z)
        zmat.append(z[t:])
        count +=1
        if count%500==0:
            print count
    zmat = np.array(zmat).T
    return zmat[0]


##-----------Steps 4-5. Guess parameter values and simulate economy-----------##

# Here are 4 different inital guesses to try.
beta0 = np.array([2.5,1.5,.5])
beta1 = np.array([1.0,3.7,1e-1])
beta2 = np.array([2.73,10.0,1e-6])
beta3 = np.array([1.59,11.04,1e-6])


# Step 6 and 7 are completed in this function
def main_SMM(beta, zmat, data_moments, t):
    """
    This is the workhorse of this file. It will use the pieces we have built so
    far to actually simulate the economy and produce MC chains for parameter
    values and moment estimations.

    Inputs:
        beta: The starting parameter value
        zmat: The matrix of shock histories created using the z_matrix function.
        data_moments: The moments we are trying to estimate, but computed
                      from the actual data.
        t: The length of each chain. Will be defined earlier.

    Returns:
        dist: A distance measure that reports how far we are from the real
              moments.
    """
    consum_sim, lab_sim, wage_sim, consum_var_sim = DSGE.DSGE_sim(beta, t, zmat)

    con_mean = np.mean(consum_sim)
    lab_mean = np.mean(lab_sim)
    wage_mean = np.mean(wage_sim)
    con_var = np.mean(consum_var_sim)

    optimal_weight = np.cov([consum_sim, lab_sim, wage_sim])

    new_mom = np.array([con_mean, lab_mean, wage_mean, con_var])
    dist = np.dot((new_mom - data_moments).T,
                   np.dot(optimal_weight, (new_mom - data_moments)))
    return dist

##-----------Step 9. Continue from step 5 until you have convergence----------##
def f(beta):
    """Just a shortcut to iterating on the main_SMM function above"""
    distance = main_SMM(beta, zmatrix, moments, t)
    print distance
    return distance

starting_x = - np.sqrt(0.004)
t = filt_consump.size

zmatrix = z_matrix(starting_x, t)
opt.fmin(f, beta0)
