"""
Created May 31, 2012

Author: Spencer Lyon
"""
import numpy as np
import scipy as sp
from numpy import sin
from scipy.signal import fftconvolve

def gen_b(k, w, want_a = 1):
    """
    This function will generate a series of b's to be used as coefficients in
    creating a band-pass filter.

    Inputs:
        k: The degree of accuracy for the bandpass filter.
        w: The frecuency about which this series of b's should be generated.

    Outputs:
        b: The b coefficients for this filter.
        a (optional, default=True): This is a 'normailzed' set of coefficients
                                    that ensures that at w=0 the filtered data
                                    will equal zero.
    """
    b = np.zeros(2*k + 1)
    ind = np.arange(1, k + 1)
    b[k] = w / np.pi
    weights = np.sin(w * ind) / (np.pi * ind)
    b[k + ind] = weights
    b[:k] = weights[::-1]

    a = b - np.mean(b)


    if want_a == 1:
        return b, a
    else:
        return b

def bandpass_filter(data, k, w1, w2):
    """
    This funciton will apply a bandpass filter to data. It will be  kth order
    and will select the band bewtween low_w and high_w.

    Inputs:
        data: The data you wish to filter
        k: The order of approximation for the filter. A max value for this is
           data.size/2
        low_w: This is the lower bound for which frecuecies will be let through.
        high_w: This is the upper bound for which frecuecies will be let through

    Outputs:
        filtered: This is the filtered data.
    """
    low_w = np.pi * 2 / w2
    high_w = np.pi * 2 / w1


    data = np.array(data)
    b_low = gen_b(k, low_w)[1]
    b_high = gen_b(k, high_w)[1]
    print b_low.shape, b_high.shape

    bweights = b_high - b_low
    y = np.empty(data.size - 2*k - 1)
    for i in range(k, y.size-k):
        y[i] = np.dot(bweights, data[i-k:i+k+1])

    return y
