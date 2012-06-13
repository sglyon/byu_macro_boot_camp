"""
Created June 8, 2012

Authors: Spencer Lyon and Chase Coleman
"""
import numpy as np
import scipy as sp
import scipy.stats as st
import pylab as pl


w = np.array([1.6907, 1.7242, 1.7552, 1.7842, 1.8113, 1.8369, 1.8610, 1.8839])
y = np.array([6, 13, 18, 28, 52, 53, 61, 60.0], dtype = float)
n = np.array([59, 60, 62, 56, 63, 59, 62, 60.0], dtype = float)

a0 = .25
b0= 4.
c0 = 2.
d0 = 10.
e0 = 2.000004
f0 = 1000.

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

    r = np.exp(r_num - r_denom)

    return r, theta_star

def main_8(theta_init, n=10000, burn=1000, call_from_other=False):
    """
    This function completes Problem 8 of the Bayes' HW set. It will do a
    simulated draw 'n' times and produce plots, give lag 1 autocorrelations,
    and report various parameters.

    Inputs:
        theta_init: The theta vector that starts off the process.
        **n: The number of iterations to be done in the algorithm.
        **burn: The number of elements in the final MC Chain that should be
                thrown out to account for the "burn-in".
        **call_from_other: This is a Boolean that says whether you are calling
                           this function from another function within the file.

    Outputs:
        theta_chain: A 3xn NumPy array representing the MC chain of theta values
                     that was generated.
        metrics: A list containing three lists each with two items:
                  1.) the lag 1 autocorrelation for each theta.
                  2.) An array with the .025, .5, .975 quantiles for each theta.

    Notes:
        ** Before input values denotes they are optional.
    """
    ## Generating the chains. The Metropolis algorithm in action.
    theta_chain = np.empty((3,n))
    theta_old = theta_init
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

    ## Plot Stuff
    hist_title = ['Histogram for $\ theta_1$',
                  'Histogram for $\ theta_2$',
                  'Histogram for $\ theta_3$']
    plot_title =['Mointoring plot for $\ theta_1$',
                 'Mointoring plot for $\ theta_2$',
                 'Mointoring plot for $\ theta_3$']

    if call_from_other==False:
        for i in range(3):
            pl.hist(theta_chain[i,:], bins = 15)
            pl.title(hist_title[i])
            pl.show()
            pl.plot(range(theta_chain[i,:].size), theta_chain[i,:])
            pl.title(plot_title[i])
            pl.show()


    ## Creating metrics.

    #Here we cut down the chain and throw out the "Burn-in" period.
    theta_chain = theta_chain[:, burn:]

    # Separating out theta's
    th1 = theta_chain[0,:]
    th2 = theta_chain[1,:]
    th3 = theta_chain[2,:]

    # Creating lag 1 autocorrelations
    corr_th1 = np.corrcoef(th1[1:], th1[:-1])[0,1]
    corr_th2 = np.corrcoef(th2[1:], th2[:-1])[0,1]
    corr_th3 = np.corrcoef(th3[1:], th3[:-1])[0,1]

    if call_from_other == False:
        print 'Correlation between th1, th2 = ', np.corrcoef(th1, th2)[0, 1]
        print 'Correlation between th1, th3 = ', np.corrcoef(th1, th3)[0, 1]
        print 'Correlation between th2, th3 = ', np.corrcoef(th2, th3)[0, 1]
        print 'acceptance rate = ', float(accept)/n


    # Computing quantiles
    end_length = n - burn
    quantiles = [round(end_length*.025),
                 round(end_length*.5),
                 round(end_length*.975)]

    quan_th1 = np.sort(th1)[quantiles]
    quan_th2 = np.sort(th2)[quantiles]
    quan_th3 = np.sort(th3)[quantiles]

    # Packaging metrics
    met_th1 = [corr_th1, quan_th1]
    met_th2 = [corr_th2, quan_th2]
    met_th3 = [corr_th3, quan_th3]

    metrics = [met_th1, met_th2, met_th3]


    return theta_chain, metrics

## Prepare new sigma matrix based on problem 8 solution.
def get_sig(theta_init):
    theta_c8, met8 = main_8(theta_init, call_from_other = True)
    theta_c8 = np.mat(theta_c8)

    th_bar = np.mat(np.mean(theta_c8, axis = 1))

    sig_tilde = 1./theta_c8.shape[1]  * (theta_c8 - th_bar) *\
                                          (theta_c8 - th_bar).T
    return sig_tilde

def main_9_input_c(theta_init, c, n=10000, burn=1000, call_from_other=False):
    """
    This function completes Problem 8 of the Bayes' HW set. It will do a
    simulated draw 'n' times and produce plots, give lag 1 autocorrelations,
    and report various parameters.

    Inputs:
        theta_init: The theta vector that starts off the process.
        c: The constant that will multiply the Sigma Covariance matrix.
        **n: The number of iterations to be done in the algorithm.
        **burn: The number of elements in the final MC Chain that should be
                thrown out to account for the "burn-in".

    Outputs:
        theta_chain: A 3xn NumPy array representing the MC chain of theta values
                     that was generated.
        metrics: A list containing three lists each with two items:
                  1.) the lag 1 autocorrelation for each theta.
                  2.) An array with the .025, .5, .975 quantiles for each theta.

    Notes:
        ** Before input values denotes they are optional.
    """

    def compute_r9(theta, sigma):
        """
        This function implements the posterior distribution as derived at the
        beginning of the problem

        Inputs:
            theta: The variable of parameters to be used in evaluating the function.
               It will be in this form:
                   theta = [mu, 1/2* log(sigma_squared), log(m1)]


        Outputs:
            r: The coefficient r computed using the equations in the problem.
            theta_star: The theta_star generated by the Jumping distribution.
        """
        theta_star = np.random.multivariate_normal(theta, sigma)

        # Taking the log before I calculate h(theta) [numerical precision errors].
        r_num = np.sum( y * np.log(g(w,theta_star)) + (n-y) * np.log((1 - g(w, theta_star)))) + \
                a0 * theta_star[2] - 2  * e0 * theta_star[1] + \
                -1/2 * ((theta_star[0] - c0)/d0)**2 - np.exp(theta_star[2])/b0 \
                        - np.exp( -2 * theta_star[1]) / f0

        r_denom = np.sum( y * np.log(g(w,theta)) + (n-y) * np.log((1 - g(w, theta)))) + \
                a0 * theta[2] - 2  * e0 * theta[1] + \
                -1/2 * ((theta[0] - c0)/d0)**2 - np.exp(theta[2])/b0 \
                        - np.exp( -2 * theta[1]) / f0

        r = np.exp(r_num - r_denom)

        return r, theta_star

    ## Define 'c' parameter that will multiply Sigma.

    sig_tilde = get_sig(theta_init)
    sig = sig_tilde * c

    ## Generating the chains. The Metropolis algorithm in action.
    theta_chain = np.empty((3,n))
    theta_old = theta_init
    theta_chain[:,0] = theta_old
    accept = 0
    test = 0
    for i in range(n):
        r, theta_star  = compute_r9(theta_old, sig)
        test = sp.rand(1)
        if r > test:
            theta_old = theta_star
            accept += 1
        else:
            theta_old = theta_old
        theta_chain[:,i] = theta_old


    ## Plot Stuff
    hist_title = ['Histogram for $\ theta_1$',
                  'Histogram for $\ theta_2$',
                  'Histogram for $\ theta_3$']
    plot_title =['Mointoring plot for $\ theta_1$',
                 'Mointoring plot for $\ theta_2$',
                 'Mointoring plot for $\ theta_3$']

    for i in range(3):
        pl.hist(theta_chain[i,:], bins = 15)
        pl.title(hist_title[i])
        pl.show()
        pl.plot(range(theta_chain[i,:].size), theta_chain[i,:])
        pl.title(plot_title[i])
        pl.show()


    ## Creating metrics.

    #Here we cut down the chain and throw out the "Burn-in" period.
    theta_chain = theta_chain[:, burn:]

    # Separating out theta's
    th1 = theta_chain[0,:]
    th2 = theta_chain[1,:]
    th3 = theta_chain[2,:]

    # Creating lag 1 autocorrelations
    corr_th1 = np.corrcoef(th1[1:], th1[:-1])[0,1]
    corr_th2 = np.corrcoef(th2[1:], th2[:-1])[0,1]
    corr_th3 = np.corrcoef(th3[1:], th3[:-1])[0,1]


    print 'Correlation between th1, th2 = ', np.corrcoef(th1, th2)[0, 1]
    print 'Correlation between th1, th3 = ', np.corrcoef(th1, th3)[0, 1]
    print 'Correlation between th2, th3 = ', np.corrcoef(th2, th3)[0, 1]
    print 'acceptance rate = ', float(accept)/n


    # Computing quantiles
    end_length = n - burn
    quantiles = [round(end_length*.025),
                 round(end_length*.5),
                 round(end_length*.975)]

    quan_th1 = np.sort(th1)[quantiles]
    quan_th2 = np.sort(th2)[quantiles]
    quan_th3 = np.sort(th3)[quantiles]

    # Packaging metrics
    met_th1 = [corr_th1, quan_th1]
    met_th2 = [corr_th2, quan_th2]
    met_th3 = [corr_th3, quan_th3]

    metrics = [met_th1, met_th2, met_th3]


    return theta_chain, metrics

def main_10_input_c(theta_init, c, n=10000, burn=1000, call_from_other=False):
    """
    This function completes Problem 8 of the Bayes' HW set. It will do a
    simulated draw 'n' times and produce plots, give lag 1 autocorrelations,
    and report various parameters.

    Inputs:
        theta_init: The theta vector that starts off the process.
        **n: The number of iterations to be done in the algorithm.
        **burn: The number of elements in the final MC Chain that should be
                thrown out to account for the "burn-in".
        **call_from_other: This is a Boolean that says whether you are calling
                           this function from another function within the file.

    Outputs:
        theta_chain: A 3xn NumPy array representing the MC chain of theta values
                     that was generated.
        metrics: A list containing three lists each with two items:
                  1.) the lag 1 autocorrelation for each theta.
                  2.) An array with the .025, .5, .975 quantiles for each theta.

    Notes:
        ** Before input values denotes they are optional.
    """


    def compute_r10(theta, sigma):
        """
        This function implements the posterior distribution as derived at the
        beginning of the problem

        Inputs:
            theta: The variable of parameters to be used in evaluating the function.
               It will be in this form:
                   theta = [mu, 1/2* log(sigma_squared), log(m1)]


        Outputs:
            r: The coefficient r computed using the equations in the problem.
            theta_star: The theta_star generated by the Jumping distribution.
        """

        jump_theta = np.array([1.8, -4.0, -1.0], dtype = float)

        theta_star = np.random.multivariate_normal(jump_theta, sigma)

        # Taking the log before I calculate h(theta) [numerical precision errors].
        r_num = np.sum( y * np.log(g(w,theta_star)) + (n-y) * np.log((1 - g(w, theta_star)))) + \
                a0 * theta_star[2] - 2  * e0 * theta_star[1] + \
                -1/2 * ((theta_star[0] - c0)/d0)**2 - np.exp(theta_star[2])/b0 \
                        - np.exp( -2 * theta_star[1]) / f0

        r_denom = np.sum( y * np.log(g(w,theta)) + (n-y) * np.log((1 - g(w, theta)))) + \
                a0 * theta[2] - 2  * e0 * theta[1] + \
                -1/2 * ((theta[0] - c0)/d0)**2 - np.exp(theta[2])/b0 \
                        - np.exp( -2 * theta[1]) / f0

        r = np.exp(r_num - r_denom)

        return r, theta_star

    sig_tilde = get_sig(theta_init)
    sig = sig_tilde * c

    ## Generating the chains. The Metropolis algorithm in action.
    theta_chain = np.empty((3,n))
    theta_old = theta_init
    theta_chain[:,0] = theta_old
    accept = 0
    for i in range(n):
        r, theta_star  = compute_r10(theta_old, sig)
        test = sp.rand(1)
        if r > test:
            theta_old = theta_star
            accept += 1
        else:
            theta_old = theta_old
        theta_chain[:,i] = theta_old

    ## Plot Stuff
    hist_title = ['Histogram for $\ theta_1$',
                  'Histogram for $\ theta_2$',
                  'Histogram for $\ theta_3$']
    plot_title =['Mointoring plot for $\ theta_1$',
                 'Mointoring plot for $\ theta_2$',
                 'Mointoring plot for $\ theta_3$']

    if call_from_other==False:
        for i in range(3):
            pl.hist(theta_chain[i,:], bins = 15)
            pl.title(hist_title[i])
            pl.show()
            pl.plot(range(theta_chain[i,:].size), theta_chain[i,:])
            pl.title(plot_title[i])
            pl.show()


    ## Creating metrics.

    #Here we cut down the chain and throw out the "Burn-in" period.
    theta_chain = theta_chain[:, burn:]

    # Separating out theta's
    th1 = theta_chain[0,:]
    th2 = theta_chain[1,:]
    th3 = theta_chain[2,:]

    # Creating lag 1 autocorrelations
    corr_th1 = np.corrcoef(th1[1:], th1[:-1])[0,1]
    corr_th2 = np.corrcoef(th2[1:], th2[:-1])[0,1]
    corr_th3 = np.corrcoef(th3[1:], th3[:-1])[0,1]

    if call_from_other == False:
        print 'Correlation between th1, th2 = ', np.corrcoef(th1, th2)[0, 1]
        print 'Correlation between th1, th3 = ', np.corrcoef(th1, th3)[0, 1]
        print 'Correlation between th2, th3 = ', np.corrcoef(th2, th3)[0, 1]
        print 'acceptance rate = ', float(accept)/n


    # Computing quantiles
    end_length = n - burn
    quantiles = [round(end_length * .025),
                 round(end_length * .5),
                 round(end_length * .975)]

    quan_th1 = np.sort(th1)[quantiles]
    quan_th2 = np.sort(th2)[quantiles]
    quan_th3 = np.sort(th3)[quantiles]

    # Packaging metrics
    met_th1 = [corr_th1, quan_th1]
    met_th2 = [corr_th2, quan_th2]
    met_th3 = [corr_th3, quan_th3]

    metrics = [met_th1, met_th2, met_th3]


    return theta_chain, metrics


def problem_12(algorithm, c_value):
    if algorithm == 9:
        main_9_input_c(theta_1, c_value)
        print 'FINISHING FIRST CHAIN'
        print '****************************\n****************************\n****************************\n****************************\n'
        main_9_input_c(theta_2, c_value)
        print 'FINISHING SECOND CHAIN'
        print '****************************\n****************************\n****************************\n****************************\n'
        main_9_input_c(theta_3, c_value)
        print 'FINISHING THIRD CHAIN'
        print '****************************\n****************************\n****************************\n****************************\n'

    elif algorithm == 10:
        main_10_input_c(theta_1, c_value)
        print 'FINISHING FIRST CHAIN'
        print '****************************\n****************************\n****************************\n****************************\n'
        main_10_input_c(theta_2, c_value)
        print 'FINISHING SECOND CHAIN'
        print '****************************\n****************************\n****************************\n****************************\n'
        main_10_input_c(theta_3, c_value)
        print 'FINISHING THIRD CHAIN'
        print '****************************\n****************************\n****************************\n****************************\n'
    else:
        print 'Error. Pick integers 9 or 10 for algorithm.'

#problem_12(9,1.)
#problem_12(9,4.)
#problem_12(9,1/4.)
#problem_12(10,1.)
#problem_12(10,4.)
#problem_12(10,1/4.)
