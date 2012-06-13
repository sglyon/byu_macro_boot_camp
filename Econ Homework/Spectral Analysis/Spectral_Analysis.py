"""
Created __, 2012

Author: Spencer Lyon
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pan
from pandas.io.data import DataReader as dr
import datetime
import hp_filter as hp
import Bandpass_Filter as bpf
from prettytable import PrettyTable as pt

# CPI: CPIAUCSL
# GDP: GDPC1
# consumption: PCECC96
# investment: GCEC96

gdp = np.asarray(
    dr('GDPC1', 'fred', start = datetime.datetime(1947,1,1))['VALUE'])
cpi = np.asarray(
    dr('CPIAUCSL', 'fred', start = datetime.datetime(1947,1,1))['VALUE'])
cons = np.asarray(
    dr('PCECC96', 'fred', start = datetime.datetime(1947,1,1))['VALUE'])
inv = np.asarray(
    dr('GCEC96', 'fred', start = datetime.datetime(1947,1,1))['VALUE'])

mask = np.arange(1,cpi.size, 3)
cpi = cpi[mask]

def problem_5():
    """
    This function uses pyplot.psd to plot the spectrum for each of the time
    series generated above.

    Inputs:
        None

    Outputs:
        This will automatically generate a plot for each of the 4 data-sets.
    """

    plt.figure()
    plt.psd(gdp)
    plt.title('GDP Spectrum')
    plt.show()

    plt.figure()
    plt.psd(cpi)
    plt.title('CPI Spectrum')
    plt.show()

    plt.figure()
    plt.psd(cons)
    plt.title('Consumption Spectrum')
    plt.show()

    plt.figure()
    plt.psd(inv)
    plt.title('Investment Spectrum')
    plt.show()

def problem_6():
    """
    This function uses the HP filter created before to filter the data
    before repeating the same process as in problem 5. We will plot both the
    trend and cycle produced by the HP filter.

    Inputs:
        None

    Outputs:
        This function automatically generates plots of the spectra for the
        filtered data.

    Notes:
        This function uses the function hp_filter found in the file hp_filter.py
    """
    gdpT, gdpC = hp.hp_filter(gdp)
    cpiT, cpiC = hp.hp_filter(cpi)
    consT, consC = hp.hp_filter(cons)
    invT, invC = hp.hp_filter(inv)

    plt.figure()
    plt.psd(gdpT)
    plt.title('GDP Trend Spectrum')
    plt.show()

    plt.figure()
    plt.psd(cpiT)
    plt.title('CPI Trend Spectrum')
    plt.show()

    plt.figure()
    plt.psd(consT)
    plt.title('Consumption Trend Spectrum')
    plt.show()

    plt.figure()
    plt.psd(invT)
    plt.title('Investment Trend Spectrum')
    plt.show()

    plt.figure()
    plt.psd(gdpC)
    plt.title('GDP Cycle Spectrum')
    plt.show()

    plt.figure()
    plt.psd(cpiC)
    plt.title('CPI Cycle Spectrum')
    plt.show()

    plt.figure()
    plt.psd(consC)
    plt.title('Consumption Cycle Spectrum')
    plt.show()

    plt.figure()
    plt.psd(invC)
    plt.title('Investment Cycle Spectrum')
    plt.show()

def problem_8():
    """
    This function produces the solution to number 8. We will keep desired
    periods at 6 and 32 but change the value of k to be [4, 8, 12, 16, 20].
    This means we will have 5 lines on the plot. We will do this for both
    filters where we force the 0 frequency to be filtered out and also when
    we don't.

    Inputs:
        None

    Outputs:
        This function automatically generates the desired plot for
        k = [4, 8, 12, 16, 20]

    """
    def BK_gain(k):
        """
        This function computes the gain for the BK filter given a particular
        value of k.

        Inputs:
            k: The degree of approximation for the filter.

        Outputs:
            a: A NumPy array for the gain at each frequency when we force the
               w = 0 frequency to be completely filtered out.
            b: A NumPy array for the gain at each frequency when we don't
               force the w = 0 frequency to be completely filtered out.
        """
        # Getting the a_h and b_h coefficients
        low_w = np.pi * 2 / 32 # was w2 in previous filter.
        high_w = np.pi * 2 / 6 # was w1 in previous filter.

        b_low, a_low = bpf.gen_b(k, low_w)
        b_high, a_high = bpf.gen_b(k, high_w)

        aweights = a_high - a_low
        bweights = b_high - b_low

        # Populating the frequency series for both a and b.
        w = np.arange(0, 2 * np.pi + .01, .01)
        h = np.arange(-k, k+1)

        a = np.array([])
        b = np.array([])

        for freq in range(0, w.size):
            a_Scalar = abs(np.dot(aweights, np.exp(1j*w[freq] * h)))
            b_Scalar = abs(np.dot(bweights, np.exp(1j*w[freq] * h)))

            a = np.append(a, a_Scalar)
            b = np.append(b, b_Scalar)

        return a, b

    w = np.arange(0, 2 * np.pi + .01, .01)
    k = np.array([4, 8, 12, 16, 20])
    k_titles = ['k = 4', 'k = 8', 'k = 12', 'k = 16', 'k = 20']

    gains_a = np.asarray([BK_gain(k[i])[0] for i in range(k.size)])
    gains_b = np.asarray([BK_gain(k[i])[1] for i in range(k.size)])

    for i in range(gains_a.shape[0]):
        plt.plot(w, gains_a[i], label = k_titles[i])
        plt.legend(loc = 0)

    plt.title('Gain for BK filter. Forcing 0 at w=0')
    plt.show()

    for i in range(gains_b.shape[0]):
        plt.plot(w, gains_b[i], label = k_titles[i])
        plt.legend(loc = 0)

    plt.title('Gain for BK filter. Not forcing 0 at w=0')
    plt.show()

    return

def problem_9():
    """
    This function produces the answer to problem 9 in the homework.
    The problem asks us to find the gain for:
        HP filter with lambda = [6.25, 1600, 129600]
        BK filter with K = [4, 8, 12, 16, 20]
        First-Difference filter

    More info for the BK filter gain can be seen in problem 8.

    The functional forms for the gain of the other filters are as follows:
        HP: 1 / (1 + 4 * lambda * (1- cos(w)))**2
        First-Diff: sqrt(2) - sqrt(1 - cos(w))

    Inputs:
        None

    Outputs:
        This function generates the plots that are asked for in the prob.
    """
    w = np.arange(0, 2 * np.pi + .01, .01)

    # Generate HP filter gains
    lamb = np.array([6.25, 1600, 129600])
    hp_labels = ['$\lambda = 6.25$', '$\lambda = 1600$', '$\lambda = 129600$']
    hps = np.asarray([1 / ((1 + 4 * lamb[i] * (1- np.cos(w))))**2
                      for i in range(lamb.size)])

    plt.figure()
    for i in range(lamb.size):
        plt.plot(w, hps[i], label = hp_labels[i])

    plt.title('HP Filter Gain')
    plt.legend(loc = 0)
    plt.show()

    # Generate First Difference Gains
    plt.figure()
    plt.plot(w, np.sqrt(2) - np.sqrt(1 - np.cos(w)))
    plt.title('First Difference Gain')
    plt.show()

    # Generate BK filter gains.
    k = np.array([4, 8, 12, 16, 20])
    k_titles = ['k = 4', 'k = 8', 'k = 12', 'k = 16', 'k = 20']

    def BK_gain(k):
        """
        This function computes the gain for the BK filter given a particular
        value of k.

        Inputs:
            k: The degree of approximation for the filter.

        Outputs:
            a: A NumPy array for the gain at each frequency when we force the
               w = 0 frequency to be completely filtered out.
            b: A NumPy array for the gain at each frequency when we don't
               force the w = 0 frequency to be completely filtered out.
        """
        # Getting the a_h and b_h coefficients
        low_w = np.pi * 2 / 32
        high_w = np.pi * 2 / 6

        b_low, a_low = bpf.gen_b(k, low_w)
        b_high, a_high = bpf.gen_b(k, high_w)

        aweights = a_high - a_low
        bweights = b_high - b_low

        # Populating the frequency series for both a and b.
        w = np.arange(0, 2 * np.pi + .01, .01)
        h = np.arange(-k, k+1)

        a = np.array([])
        b = np.array([])

        for freq in range(0, w.size):
            a_Scalar = abs(np.dot(aweights, np.exp(1j*w[freq] * h)))
            b_Scalar = abs(np.dot(bweights, np.exp(1j*w[freq] * h)))

            a = np.append(a, a_Scalar)
            b = np.append(b, b_Scalar)

        return a, b

    gains_a = np.asarray([BK_gain(k[i])[0] for i in range(k.size)])

    for i in range(gains_a.shape[0]):
        plt.plot(w, gains_a[i], label = k_titles[i])
        plt.legend(loc = 0)

    plt.title('Gain for BK filter. Forcing 0 at w=0')
    plt.show()

def problem_10():
    """
    This function provides the solution to problem 10. We use the 4 data
    sets from problem 5 to generate moments regarding data filtered with
    an HP filter, BK filter, and first-difference filter.

    Inputs:
        None

    Outputs:
        the_tables: This is the tables containing mean, standard deviation,
                    autocorrelation, and correlation with GDP for the
                    following filters for each data set:
                        HP(1600)
                        BK(16, 6, 32)
                        first difference.
    """
    # Generate GDP filtered data
    first_diff_gdp = gdp[1:] - gdp[:-1]
    hp_gdp = hp.hp_filter(gdp)[1]
    bp_gdp = bpf.bandpass_filter(gdp, 16, 6, 32)[16:-16]
    all_gdp = np.array([first_diff_gdp, hp_gdp, bp_gdp])

    # Generate cpi filtered data
    first_diff_cpi = cpi[1:] - cpi[:-1]
    hp_cpi = hp.hp_filter(cpi)[1]
    bp_cpi = bpf.bandpass_filter(cpi, 16, 6, 32)[16:-16]
    all_cpi = np.array([first_diff_cpi, hp_cpi, bp_cpi])

    # Generate consumption filtered data
    first_diff_cons = cons[1:] - cons[:-1]
    hp_cons = hp.hp_filter(cons)[1]
    bp_cons = bpf.bandpass_filter(cons, 16, 6, 32)[16:-16]
    all_cons = np.array([first_diff_cons, hp_cons, bp_cons])

    # Generate investment filtered data
    first_diff_inv = inv[1:] - inv[:-1]
    hp_inv = hp.hp_filter(inv)[1]
    bp_inv = bpf.bandpass_filter(inv, 16, 6, 32)[16:-16]
    all_inv = np.array([first_diff_inv, hp_inv, bp_inv])

    all_data = np.vstack([all_gdp, all_cpi, all_cons, all_inv])


    all_tables = np.empty((4, 4, 3))

    data_titles = ['GDP', "CPI", 'Consumption', 'Investment']
    filter_labels = ['First Diff', 'HP', 'BP']


    for data_set in range(all_data.shape[0]):
        plt.figure()
        for filt in range(all_data.shape[1]):
            plt.plot(
                range(all_data[data_set, filt].size),
                all_data[data_set,filt],
                label = filter_labels[filt])
        plt.title(data_titles[data_set])
        plt.legend(loc=0)
        plt.show()


    for data_set in range(all_data.shape[0]):
        for i in range(all_data.shape[1]):
            all_tables[data_set, 0, i] = np.mean(all_data[data_set, i])
            all_tables[data_set, 1, i] = np.std(all_data[data_set, i])
            all_tables[data_set, 2, i] = np.corrcoef(all_data[data_set, i][:-1],
                                                all_data[data_set, i][1:])[0,1]
            all_tables[data_set, 3, i] = np.corrcoef(all_data[data_set, i],
                                                     all_data[0, i])[0,1]

    titles = ['Parameters [GDP]', 'Parameters [CPI]',
              'Parameters [cons]', 'Parameters [inv]']
    col_names = ['First Diff', 'HP (1600)', 'BK (16)']
    params = ['Mean', 'Std', 'Autocorrelation', 'Corr w/GDP']

    pretty_gdp = pt()
    pretty_cpi = pt()
    pretty_cons = pt()
    pretty_inv = pt()

    pretty_all = [pretty_gdp, pretty_cpi, pretty_cons, pretty_inv]

    for tab in range(4):
        pretty_all[tab].add_column(titles[tab], params)
        for col in range(3):
            pretty_all[tab].add_column(col_names[col], all_tables[tab, :, col])
        print pretty_all[tab]

    return pretty_all

problem_10()
