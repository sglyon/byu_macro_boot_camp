"""
Created on MAy 14, 2012

Author: Spencer Lyon
"""

import math
import scipy as sp

# Problem 1
def myFactorial(n):
    """
    This function uses recursive mehtods to compute n!.

    Inputs:
        n: The number for which you want to compute the factorial

    Outputs:
        result: The factorial of the input n
    """
    if n == 0:
        return 1

    result = n * myFactorial(n-1)
    print result

    return result


# Problem 2
def their_gcd(a,b):
    """
    Uses recursive methods to find the greatest common denominator between
    two numbers

    Inputs:
        a: The first number to use in finding the gcd. Must be larger than b.
        b: The second number to use in finding the gcd. Must be smaller than a.

    Outputs:
        result: The gcd between the two numbers.
    """
    assert a >= b and b >= 0

    if b == 0:
        return a
    else:
        return their_gcd(b,a % b)


def my_gcd(a,b):
    """
    This function computes the greatest common denominator bewteen two numbers.

    Inputs:
        a: The first number to use in finding the gcd.
        b: The second number to use in finding the gcd.

    Outputs:
        result: The greatest common denominator between the two numbers.
    """
    assert a > b and b >= 0

    while b > 0:
        tem = b
        b = a % b
        a = tem

    return a



# Problem 3
def laplaceDet(A):
    """
    Calcualtes the determinant of a matrix A using Laplace's equation as a
    recursive implementation of cofactor expansion.

    Inputs:
        A: The matrix for which you want to find the determinant.

    Outputs:
        result: The determinant of the matrix A.
    """

    n = A.shape[0]
    result = 0

    # Base case
    if n ==2:
        return A[0,0]*A[1,1] - A[0,1]*A[1,0]

    for i in range(0,n):
        result += A[0,i] *laplaceDet(sp.hstack((A[1:n, 0:i], A[1:n, i+1:n])))\
                *(-1)**i

    return result
