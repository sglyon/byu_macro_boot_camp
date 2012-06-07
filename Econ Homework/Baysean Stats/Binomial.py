import math
import scipy as sp

def x_choose_y(n,k):
    """
    This is a quick implementation of the n choose k function

    Inputs:
        n: How many total options there are
        k: How many you want to choose out of the n.

    Outputs:
        The value n choose k
    """
    return math.factorial(n)/(math.factorial(k) * math.factorial(n-k))

def binomial_pdf(k, n, p):
    """
    This function computes the value of the binomial pdf given parameters
    n, k, and p.

    Inputs:
        k: The number of sucesses you are going after.
        n: Total sample size.
        p: Estimated probability.

    Outputs:
        The probability of getting k successes.
    """
    return x_choose_y(n,k) * p**k * (1-p) ** (n-k)

def binomial_cdf(k, n, p):
    """
    Returns the cdf of the binomial distribution given parameters n and p as
    well as a number of sucesses x.

    Inputs:
        k: The number of sucesses you are going after.
        n: Total sample size.
        p: Estimated probability.

    Outputs:
        The cdf associated with getting x sucesses.
    """
    pdf_list = [binomial_pdf(i, n, p) for i in range(0,k+1)]
    return sp.sum(pdf_list)
