"""
Created May 30, 2012

Author: Spencer Lyon

"""
import numpy as np
import scipy as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la
import matplotlib.pyplot as plt
import pylab
import Bandpass_Filter as bpf
import hp_filter as hpf

sp.set_printoptions(linewidth=140, suppress = True, precision = 5)

def problem_1():
    """
    This function uses the file GPDIC96.txt as the data for this problem.
    It produces the T, cycle sets and plots what Ben asked for
    """
    data = sp.loadtxt('GPDIC96.txt', skiprows=1, dtype=float,
                      converters={0:pylab.datestr2num})
    yData = data[:,1]
    T, c  = hpf.hp_filter(yData)
    time = sp.linspace(0, T.size, T.size)
    plt.plot(time, yData)
    plt.plot(time, T)
    plt.figure()
    plt.plot(time, c)
    plt.show()

def problem_2():
    """
    This function completes problem two from Ben's assignment
    """
    data = sp.loadtxt('GPDIC96.txt', skiprows=1, dtype=float,
                      converters={0:pylab.datestr2num})
    yData = data[:,1]
    T_1000, c_1000  = hpf.hp_filter(yData, 1000)
    T_1600, c_1600  = hpf.hp_filter(yData, 1600)
    T_1800, c_1800  = hpf.hp_filter(yData, 1800)
    T_16000, c_16000  = hpf.hp_filter(yData, 16000)

    all_c = np.array([c_1000, c_1600, c_1800, c_16000])

    the_table = np.zeros((3,all_c.shape[0]))

    for i in range(all_c.shape[0]):
        the_table[0, i] = np.mean(all_c[i])
        the_table[1, i] = np.var(all_c[i])
        the_table[2, i] = np.corrcoef(all_c[i][:-1], all_c[i][1:])[0,1]

    return the_table

def problem_3():
    """
    This function competes problem three from Ben's assignment.
    """
    data = sp.loadtxt('SP500.txt', skiprows=1, dtype=float,
                      converters={0:pylab.datestr2num})
    yData = data[:,1]

    # Differencing (y_t - y_t-1)
    ytemp = yData.copy()
    ytemp2 = yData.copy()
    ytemp2 = np.insert(ytemp2, 0,0)
    ytemp = np.insert(ytemp, ytemp.size, 0)
    yhat = ytemp - ytemp2
    c_diff = yhat[1:-1]

    # OLS (quadratic)
    x = np.zeros((yData.size,3))
    x[:, 0] = np.arange(1, yData.size + 1)**2
    x[:, 1] = np.arange(1, yData.size + 1)
    x[:, 2] = np.ones(yData.size)
    beta = la.lstsq(x, yData)[0]

    time = np.arange(1, yData.size+1 )
    y_trend = beta[0]*time**2 + beta[1] * time + beta[2] * np.ones(time.size)
    c_ols = yData - y_trend


    # HP filter
    T_hp, c_hp = hpf.hp_filter(yData, 1600)

    # Band-Pass
    c_bp = bpf.bandpass_filter(yData, 6, 6, 32)

    all_c = np.array([c_diff, c_ols, c_hp, c_bp])
    the_table = np.zeros((3,all_c.shape[0]))

    for i in range(all_c.shape[0]):
        the_table[0, i] = np.mean(all_c[i])
        the_table[1, i] = np.var(all_c[i])
        the_table[2, i] = np.corrcoef(all_c[i][:-1], all_c[i][1:])[0,1]

    return the_table

def problem_4():
    """
    This function competes problem three from Ben's assignment.
    """
    data = sp.loadtxt('SP500.txt', skiprows=1, dtype=float,
                      converters={0:pylab.datestr2num})
    yData = data[:,1]

    # Differencing (y_t - y_t-1)
    ytemp = yData.copy()
    ytemp2 = yData.copy()
    ytemp2 = np.insert(ytemp2, 0,0)
    ytemp = np.insert(ytemp, ytemp.size, 0)
    yhat = ytemp - ytemp2
    c_diff = yhat[1:-1]
    five_percent_diff = round(yhat.size * .05)
    c_diff = c_diff[five_percent_diff:-five_percent_diff]

    # OLS (quadratic)
    x = np.zeros((yData.size,3))
    x[:, 0] = np.arange(1, yData.size + 1)**2
    x[:, 1] = np.arange(1, yData.size + 1)
    x[:, 2] = np.ones(yData.size)
    beta = la.lstsq(x, yData)[0]
    time = np.arange(1, yData.size+1 )
    y_trend = beta[0]*time**2 + beta[1] * time + beta[2] * np.ones(time.size)

    c_ols = yData - y_trend
    five_percent_ols = round(c_ols.size * .05)

    c_ols  = c_ols[five_percent_ols:-five_percent_ols]


    # HP filter
    T_hp, c_hp = hpf.hp_filter(yData, 1600)
    five_percent_hp = round(c_hp.size * .05)
    c_hp  = c_hp[five_percent_hp:-five_percent_hp]

    # Band-Pass
    c_bp = bpf.bandpass_filter(yData, 6, 6, 32)
    five_percent_bp = round(c_bp.size * .05)
    c_bp  = c_bp[five_percent_bp:-five_percent_bp]

    all_c = np.array([c_diff, c_ols, c_hp, c_bp])
    the_table = np.zeros((3,all_c.shape[0]))

    for i in range(all_c.shape[0]):
        the_table[0, i] = np.mean(all_c[i])
        the_table[1, i] = np.var(all_c[i])
        the_table[2, i] = np.corrcoef(all_c[i][:-1], all_c[i][1:])[0,1]

    return the_table
