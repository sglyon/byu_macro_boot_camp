"""
Created June 14, 2012

Author: Spencer Lyon
"""
import numpy as np
from sympy import symbols, nsolve
import numpy.linalg as npla
import matplotlib.pyplot as plt
import scipy as sp

nans, GDP, Unem = np.genfromtxt('RGDPU.txt', skip_header=1,unpack=True)

# We already have log GDP. Now we need the difference in log(rGDP)
GDP_rate = GDP[:-1] - GDP[1:]

# We took all but the last entry in Unem to preserve dimensions among it and GDP
y = np.vstack((GDP_rate, Unem[0:-1]))

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
Beta = np.dot(np.dot(no_lag, X.T), npla.inv(np.dot(X, X.T)))
bet = Beta[:,1:]
B1, B2, B3, B4, B5, B6, B7, B8 =\
        [np.mat(bet[:,0:2]), np.mat(bet[:,2:4]), np.mat(bet[:,4:6]),
         np.mat(bet[:,6:8]), np.mat(bet[:,8:10]), np.mat(bet[:,10:12]),
         np.mat(bet[:,12:14]), np.mat(bet[:,14:16])]

# We now find the predicted y values so we can find the residuals.
Y_hat = np.dot(Beta, X)
resid = no_lag - Y_hat
sigma = np.cov(resid) # This is the sigma we will use to find A_0 inverse.


## Solve for D(1) and then C(1)
# Solve for D(1)
B_1 = np.eye(2) - B1 - B2 - B3 - B4 - B5 - B6 - B7 - B8
D_1 = npla.inv(B_1)

#Solve symbolically for C_1
c11, c21, c22 = symbols(['c1', 'c2', 'c3'])
C_12 =  np.dot(np.dot(D_1, sigma),D_1.T)
g1 = c11**2 - C_12[0,0]
g2 = c11*c21 - C_12[1,0]
g3 = c21**2 + c22**2 - C_12[1,1]
c_1_sol =  nsolve((g1,g2,g3),(c11,c21,c22),(-.03,.113,-.3))
C_1 = np.empty([2,2])
C_1[0,0] = c_1_sol[0]
C_1[1,0] = c_1_sol[1]
C_1[1,1] = c_1_sol[2]


## Compute A_0_inv and A_0
A_0_inv = np.dot(npla.inv(D_1), C_1)
A_0 = npla.inv(A_0_inv)

## Create D matricies
D0 = np.mat(np.eye(n_vars))
D1 = B1*D0
D2 = B1*D1 + B2*D0
D3 = B1*D2 + B2*D1 + B3*D0
D4 = B1*D3 + B2*D2 + B3*D1 + B4*D0
D5 = B1*D4 + B2*D3 + B3*D2 + B4*D1 + B5*D0
D6 = B1*D5 + B2*D4 + B3*D3 + B4*D2 + B5*D1 + B6*D0
D7 = B1*D6 + B2*D5 + B3*D4 + B4*D3 + B5*D2 + B6*D1 + B7*D0
D8 = B1*D7 + B2*D6 + B3*D5 + B4*D4 + B5*D3 + B6*D2 + B7*D1 + B8*D0
big_D = [D0, D1, D2, D3, D4, D5, D6, D7, D8]

for i in range(8,40):
    Dnew = B1*big_D[-1] + B2*big_D[-2] + B3*big_D[-3] + B4*big_D[-4] + \
           B5*big_D[-5] + B6*big_D[-6] + B7*big_D[-7] + B8*big_D[-8]
    big_D.append(Dnew)


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
    C = np.dot(big_D[i], A_0_inv)
    big_C.append(C)

"""TODO: Fix IRFs and plot them"""

c_array = np.array(big_C)

IRF_U_AS = np.hstack((0,np.cumsum(c_array[:,0,0])))
IRF_U_AD = np.hstack((0,np.cumsum(c_array[:,0,1])))
IRF_GDP_AS = np.hstack((0, c_array[:,1,0]))
IRF_GDP_AD = np.hstack((0, c_array[:,1,1]))

x = np.linspace(0,41,42)

plt.figure()
plt.title('Long run IRF of Unemployment to Aggregate Supply Shock')
plt.plot(range(IRF_U_AS.size), IRF_U_AS)
plt.show()

plt.figure()
plt.title('Long run IRF of Unemployment to Aggregate Demand Shock')
plt.plot(range(IRF_U_AD.size), IRF_U_AD)
plt.show()

plt.figure()
plt.title('Long run IRF of GDP to Aggregate Supply Shock')
plt.plot(range(IRF_GDP_AS.size), IRF_GDP_AS)
plt.show()

plt.figure()
plt.title('Long run IRF of GDP to Aggregate Demand Shock')
plt.plot(range(IRF_GDP_AD.size), IRF_GDP_AD)
plt.show()
