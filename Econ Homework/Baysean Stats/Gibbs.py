"""
Created June 5, 2012

Author: Spencer Lyon
hi
"""
import numpy as np
import scipy as sp
import scipy.stats as st
import pylab
from prettytable import PrettyTable as pt

sp.set_printoptions(linewidth=140, suppress = True, precision = 5)

i = sp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = sp.array([5, 1, 5, 14, 3, 19, 1, 1, 4, 22])
t = sp.array([94.320, 15.720, 62.880, 125.76, 5.24,
			  31.44, 1.048, 1.048, 2.096, 10.48])
r = sp.array([.053, .064, .080, .111, .573, .604, .954, .954, 1.91, 2.099])


to_pass = np.vstack((y, t))

def gibbs(obs, n=10000, burn=1000):
    """
    This function applies the Gibb's Sampling algorithm to the data passed in.
    By default we will take 10000 monte carlo samples, drop the first 1000 (for
    the burn in) and then take every 200th entry after that. That leaves us with
    a sample with 95 entries.

    We will assume that for this model that we have the following distributions:
        liklihood: f(y|theta) ~P(theta)
        Prior: g(theta|beta) ~ G(alpha, beta)
        Hyperprior: h(beta) ~ invGamma (c, d)

    Inputs:
        n: The number of elements to be generate in the MC process
        burn: The number of elements ot be thrown out to account for burn-in
        obs: The data set over which you want to do the Gibbs sampling.

    Outputs:
        bet: The (n-burn)x1 estimated beta chain.
        thet: The (n-burn)x3 estiamted theta chain.
    """
    alpha = 0.7
    d = 1.0
    c = 0.1

    k = obs.shape[1]

    beta = sp.rand(1) * 5
    theta = sp.rand(k) * .5

    beta_chain = np.empty((n,1))
    theta_chain = np.empty((n,k))

    for i in range(1,n):
        beta_chain[i] = st.invgamma.rvs(k * alpha + c,
                                scale = (1 / d + np.sum(theta_chain[i-1,:])))
        for j in range(k):
            theta_chain[i,j] = st.gamma.rvs(obs[0,j] + alpha,
                                        scale= 1 /(obs[1,j] + 1/beta_chain[i]))

    picky = np.arange(0,n-burn, 100)
    beta_chain = beta_chain[burn:]
    theta_chain = theta_chain[burn:, :]
    bet = beta_chain[picky]
    thet = theta_chain[picky,:]

    return [bet, thet]

def moments(beta, theta):
    """
    This funciton takes the output from the gibbs function and computes the
    95% credible interval and median for each chain.

    Inputs:
        beta: this is an nx1 vector of values for beta.
        theta: This is an nx3 vector of values for each of the 3 theta params.

    Returns:
        Just prints out the desired moments.
    """
    column_titles = ['Beta', "Theta_1", 'Theta_2', 'Theta_3',
                     "Theta_4", 'Theta_5', 'Theta_6',
                     "Theta_7", 'Theta_8', 'Theta_9', 'Theta_10']

    all_params = np.hstack((beta, theta))
    all_sort = np.sort(all_params, axis=1)
    items = all_sort[:,0].size
    chop = round(items * 0.025)

    table = np.empty((2,11))
    for param in range(all_params.shape[1]):
        print column_titles[param], 'Mean', np.mean(all_params[:,param])
        print column_titles[param], 'Median', np.median(all_params[:,param])
        print column_titles[param], 'Standard Deviation',\
                                    np.std(all_params[:,param])

        print column_titles[param] ,' Credible Set',\
                            [all_sort[chop, param], all_sort[-chop, param]], '\n'

def gen_plots():
    """
    This function generates a quick plot of the beta MC chain as well as the
    a histogram of its distribution. It does the same for each theta.

    Inputs:
        None

    Outputs:
        None: This funciton automaticlly generates and shows all plots.
    """

    beta, theta = gibbs(to_pass)

    pylab.hist(beta, bins = 40)
    pylab.title('Beta histogram')
    pylab.show()
    pylab.plot(range(beta.size), beta)
    pylab.title('Beta chain')
    pylab.show()


    titles = ["Theta_1", 'Theta_2', 'Theta_3',
                     "Theta_4", 'Theta_5', 'Theta_6',
                     "Theta_7", 'Theta_8', 'Theta_9', 'Theta_10']

    for theta_ind in range(10):
        pylab.hist(theta[:, theta_ind], bins = 40)
        pylab.title('Histogram for ' + titles[theta_ind])
        pylab.show()
        pylab.plot(range(theta[:, theta_ind].size), theta[:,theta_ind])
        pylab.title('Chain for ' + titles[theta_ind])
        pylab.show()

    return
