"""
Created June 14, 2012

Author: Spencer Lyon
"""
import numpy as np
from sympy import symbols, nsolve
import numpy.linalg as npla
import scipy as sp
import scipy.stats as st

nans, GDP, Unem = np.genfromtxt('RGDPU.txt', skip_header=1,unpack=True)

# We already have log GDP. Now we need the difference in log(rGDP)
GDP_rate = GDP[:-1] - GDP[1:]

# We took all but the last entry in Unem to preserve dimensions among it and GDP
y = np.vstack((Unem[0:-1], GDP_rate))

def lag_op(y, nlag):
    """
    This function implements the forward lag operator.

    Inputs:
        y: The mxn data set you wish to lag. M represents how many variables
           you have and n represents how many oberservations per variable.
        nlag: The number of lags you want to implement.

    Outputs:
        y_lag: The lagged data. We will go from 0 lags to n lags and stack each
               one on top of the other. So y_lag be m*nlag x (n-nlag)
    """
    # Extract first n-nlag elements of the data.
    y_lag = y[:,1:-(nlag-1)]

    for l in range(2, nlag):
        y_new = y[:,l:-(nlag-l)]
        y_lag = np.vstack((y_lag, y_new))
    y_lag = np.vstack((y_lag, y[:,nlag:]))

    return y_lag

no_lag = y[:,0:-8]
lagged = lag_op(y, 8)
n_vars, t = no_lag.shape
const_row = np.ones(t)

## Do the OLS. to get the B matricies and residuals.
X = np.vstack([const_row, lagged])

# This beta will be 2 x 17. The first column is the constant terms.
Beta = npla.lstsq(X.T, no_lag.T)[0].T

# We now find the predicted y values so we can find the residuals.
Y_hat = np.dot(X.T, Beta.T).T
resid = no_lag - Y_hat
sigma = np.cov(resid) # This is the sigma we will use to find A_0 inverse.

## Solving for A_0_inv
# Applying short run restrictions means that A_0 inv. will have no (1,2) entry.
# We will be left with sigma = [[a11**2, 0], [a11*a22, a21**2 + a22**2]]
# We solve this equation symbolically below.
a11, a21, a22 = symbols(['a11', 'a21', 'a22'])
f1 = a11**2 - sigma[0,0]
f2 = a11*a21 - sigma[1,0]
f3 = a21**2 + a22**2 - sigma[1,1]
AoInvSol = nsolve((f1,f2,f3),(a11,a21,a22),(.5,-.5,.5))
A_0_Inv = np.zeros((2,2))
A_0_Inv[0,0] = AoInvSol[0]
A_0_Inv[1,0] = AoInvSol[1]
A_0_Inv[1,1] = AoInvSol[2]
A_0 = npla.inv(A_0_Inv)
A_0[0,1] = 0.0

## Solving for D matricies
# Stack betas for easy D creation
bet = Beta[:,1:]
betas = [np.mat(bet[:,0:2]), np.mat(bet[:,2:4]), np.mat(bet[:,4:6]),
         np.mat(bet[:,6:8]), np.mat(bet[:,8:10]), np.mat(bet[:,10:12]),
         np.mat(bet[:,12:14]), np.mat(bet[:,14:16])]


## Create D matricies
D0 = np.mat(np.eye(n_vars))
D1 = betas[0]*D0
D2 = betas[0]*D1 + betas[1]*D0
D3 = betas[0]*D2 + betas[1]*D1 + betas[2]*D0
D4 = betas[0]*D3 + betas[1]*D2 + betas[2]*D1 + betas[3]*D0
D5 = betas[0]*D4 + betas[1]*D3 + betas[2]*D2 + betas[3]*D1 + betas[4]*D0
D6 = betas[0]*D5 + betas[1]*D4 + betas[2]*D3 + betas[3]*D2 + betas[4]*D1\
                 + betas[5]*D0
D7 = betas[0]*D6 + betas[1]*D5 + betas[2]*D4 + betas[3]*D3 + betas[4]*D2\
                 + betas[5]*D1 + betas[6]*D0
D8 = betas[0]*D7 + betas[1]*D6 + betas[2]*D5 + betas[3]*D4 + betas[4]*D3\
                 + betas[5]*D2 + betas[6]*D1 + betas[7]*D0

big_D = [D0, D1, D2, D3, D4, D5, D6, D7, D8]

## Get the epsilon_t by doing A_0 * resid_t
all_eps = []
for i in range(t):
    temp_resid = resid[:,i]
    eps = sp.dot(A_0, temp_resid)
    all_eps.append(eps)

## Use the epsilon_t to get C matricies. They are defined as C_i = D_i * A_0_inv
c_ind = np.shape(big_D)[0]
big_C = []
for i in range(c_ind):
    C = np.dot(big_D[i], A_0_Inv)
    big_C.append(C)

"""TODO: Fix IRFs and plot them"""


