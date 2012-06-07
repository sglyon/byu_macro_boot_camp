"""
Created May 24, 2012

Author: Spencer Lyon
"""
import numpy as np
import pylab
import scipy.stats as st
import matplotlib.pyplot as plt

# Use log-normal, gamma, or exponential as a liklihood function.
# bpost = b_prior/(n*b_prior +1)
# a_post = sum(data)+ a_prior

data = np.loadtxt('motorcycle.txt', skiprows=1, dtype = float,
                  converters = {0: pylab.datestr2num})

total = data[:,1]
merch = data[:,2]
bike =  data[:,3]
repair =  data[:,4]

sorts = data[data[:,5].argsort()]

no_sale_count = np.nonzero(sorts[:,-1])[0][0]

no_sale = sorts[0:no_sale_count,:]

yes_sale = sorts[no_sale_count:,:]

def histogram():
    """
    This function generates the histogrms associated with the data
    and asked for in the problem.

    There are no inputs or outputs. The plots are automatically created.

    """

    titles_sale = ['total sale-day',  'merch sale-day', 'bikes sale-day',
                   'repairs sale-day']

    titles_no_sale = ['total non sale-day',  'merch non sale-day',
                      'bikes non sale-day', 'repairs non sale-day']

    colors = ['hotpink', 'chartreuse', 'mediumaquamarine', 'darkorchid']


    sorts = data[data[:,5].argsort()]

    no_sale_count = np.nonzero(sorts[:,-1])[0][0]

    no_sale = sorts[0:no_sale_count,:]

    yes_sale = sorts[no_sale_count:,:]

    for i in range(1, yes_sale.shape[1]-1):
        pylab.figure()
        pylab.title(titles_sale[i-1])
        pylab.hist(yes_sale[:,i], bins = 30, normed = True, color = colors[i-1])

    pylab.show()

    for i in range(1, no_sale.shape[1]-1):
        pylab.figure()
        pylab.title(titles_no_sale[i-1])
        pylab.hist(no_sale[:,i], bins = 30, normed = True, color = colors[i-1])


# For part b we chose an exponential likelihood with a gamma prior for all the
# data sets. This makes sense because the histograms all look like they could
# be represented by those distributions.

def gen_prior_params(data_set):
    """
    This function calls scipy.stats.gamma.fit to generate parameters
    for the prior distriubtions in problem 5.

    Inputs:
        data_set: The data for which you want the prior parameters.

    Outputs:
        a: The shape parameter for the gamma distribution
        l: The location of the gamma distribution
        b: The scale parameter of the gamma distribution.
    """

    a, l, b = st.gamma.fit(data_set)

    return a, l, b

def update_prior(data_set_index):
    """
    Updates the data set to obtain a posterior distribution. The likihood is
    assumed to be an exponential distribution. The prior is a gamma(alpha, beta)
    and that produces a gamma(alpha + n, beta + n*mean(x))

    Inputs:
        data_set(int): The data for which you want the posterior distribution.
            1.) total  2.) merch  3.) bike  4.) repair

    Outputs:
        sale: The shape and scale parameters for the sale days.
        no_sale: The shape and scale parameters for the no-sale days.
    """
    data = sorts[:,data_set_index]
    no_sale = data[0:no_sale_count]
    #no_sale = data[0:10]
    yessale = data[no_sale_count:]
    #yessale = data[no_sale_count:no_sale_count + 10]


    # Get prior/posterior parameters for sale days
    pri_a_sale, xx, pri_b_sale = gen_prior_params(yessale)
    post_a_sale = pri_a_sale + yessale.size
    post_b_sale = pri_b_sale + np.sum(yessale)

    # Get prior/posterior parameters for no sale days
    pri_a_no_sale, xxx, pri_b_no_sale = gen_prior_params(no_sale)
    post_a_no_sale = pri_a_sale + no_sale.size
    post_b_no_sale = pri_b_no_sale + np.sum(no_sale)


    #get pdf values for sale, no_sale, and the difference bewteen the two.
    the_range = np.linspace(0, np.max(data)*1.05, 250)
    no_sale = st.invgamma.pdf(the_range, post_a_no_sale, scale = 1/post_b_no_sale )
    pylab.plot(the_range, no_sale, linewidth = .5)
    #pylab.hist(yes_sale[:,1], bins = 30,normed = True)
    pylab.show()
