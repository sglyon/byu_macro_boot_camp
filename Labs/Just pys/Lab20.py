"""
Created May 14, 2012

Author: Spencer Lyon

STILL TURN IN 17, 18, 20, 24, 26
"""
import scipy as sp

# Problem 1
def numerical_derivative(func, order, kind, accuracy, point, tol=.00001):
    """
    This function calculates the numerical derivative of func.

    Inputs:
        func: The function for which you want to find the derivative
        order: The order of derivative you wish to calculate
        kind: This is the kind of algorithm to be used in computing the
              derivative. This must be a string containing one of the following
              three words:
                centered: use the centered algorithm
                forward: use the forward algorithm
                backward: use the backward algorithm
        accuracy: This is the degree of accuracy to be used in the approximation
                  For centered kind this must be 2, 4, or 6
        point: The point about which to take the derivative
                  For forward or backward kind this must be 1, 2, or 3.
        tol (optional): A user-defined tolerance level. If none is given a
                        default value of 0.00001 is used.

    Outputs: the numerical derivative
    """
    deriv = []

    if order == 1:
        if kind == 'centered':
            if accuracy == 2:
               deriv.append((func(point + tol) - func(point- tol))/( 2* tol))
            elif accuracy == 4:
                deriv.append(func(point - 2 * tol)/(12 * tol) +
                             -2 * func(point  - tol)/(tol * 3) +
                             2 * func(point + tol)/ (tol * 3) +
                             func(point + 2 * tol)/(-12* tol))
            elif accuracy == 6:
                deriv.append(func(point - 3 * tol)/(-60 * tol) +
                             func(point - 2 * tol)/(20 * tol) +
                             -func(point - 1 * tol)/(2 * tol) +
                             func(point + 1 * tol)/(2 * tol) +
                             -func(point + 2 * tol)/(20 * tol) +
                             func(point + 3 * tol)/(60 * tol))
            else:
                print "Wrong accuracy. Use 2, 4, or 6 for kind 'centered'"
        elif kind == 'forward':
            if accuracy == 1:
                deriv.append(-func(point)/tol+
                             func(point + tol)/tol)
            elif accuracy == 2:
                deriv.append( - 3 * func(point)/(2 * tol)+
                             2 * func(point + tol)/ tol +
                             - func(point + 2 * tol)/ (tol * 2))
            elif accuracy ==3:
                deriv.append(-11 * func(point)/(6 * tol) +
                             3 * func(point + tol)/ tol +
                             -3 * func(point + 2 * tol)/ (tol * 2) +
                             func(point + 3 * tol)/ (tol * 3))
            else:
                print "Wrong accuracy. Use 1, 2, or 3 for kind 'forward'"
        elif kind == "backward":
            if accuracy == 1:
                deriv.append(func(point)/tol+
                             -func(point - tol)/tol)
            elif accuracy == 2:
                deriv.append( 3 * func(point)/(2 * tol)+
                             -2 * func(point - tol)/ tol +
                             func(point - 2 * tol)/ (tol * 2))
            elif accuracy ==3:
                deriv.append(11 * func(point)/(6 * tol) +
                             -3 * func(point - tol)/ tol +
                             3 * func(point - 2 * tol)/ (tol * 2) +
                             func(point - tol)/ (tol * 3))
            else:
                print "Wrong accuracy. Use 1, 2, or 3 for kind 'backward'"
    elif order ==2:
        if kind == 'centered':
            if accuracy == 2:
               deriv.append(func(point - tol)/tol+
                            -2 * func(point)/tol +
                            func(point + tol)/tol)
            elif accuracy == 4:
                deriv.append(-func(point - 2 * tol)/(12 * tol) +
                             4 * func(point  - tol)/(tol * 3) +
                             -5 * func(point)/(2* tol) +
                             4 * func(point + tol)/ (tol * 3) +
                             -func(point + 2 * tol)/(12* tol))
            elif accuracy == 6:
                deriv.append(func(point - 3 * tol)/(90 * tol) +
                             -3 * func(point - 2 * tol)/(20 * tol) +
                             3 * func(point - 1 * tol)/(2 * tol) +
                             - 49 *  func(point)/(18 * tol) +
                             3 * func(point + 1 * tol)/(2 * tol) +
                             - 3* func(point + 2 * tol)/(20 * tol) +
                             func(point + 3 * tol)/(90 * tol))
            else:
                print "Wrong accuracy. Use 2, 4, or 6 for kind 'centered'"
        elif kind == 'forward':
            if accuracy == 1:
                deriv.append(func(point)/tol+
                             -2 * func(point + tol)/tol +
                             func(point + 2 * tol)/tol)
            elif accuracy == 2:
                deriv.append(2 * func(point)/tol+
                             -5 * func(point + tol)/ tol +
                             4 * func(point + 2 *  tol)/ tol +
                             -func(point + 3 * tol)/ tol)
            elif accuracy ==3:
                deriv.append(35 * func(point)/(12 * tol) +
                             -26 * func(point + tol)/(3 * tol) +
                             19 * func(point + 2 * tol)/(2 * tol) +
                             -14 * func(point + 3 * tol)/ (tol * 3) +
                             11* func(point + 4 * tol)/ (tol * 12))
            else:
                print "Wrong accuracy. Use 1, 2, or 3 for kind 'forward'"
        elif kind == 'backward':
            if accuracy == 1:
                deriv.append(func(point)/tol+
                             2 * func(point - tol)/tol +
                             -func(point - 2 * tol)/tol)
            elif accuracy == 2:
                deriv.append(-2 * func(point)/tol+
                             5 * func(point - tol)/ tol +
                             -4 * func(point - 2 *  tol)/ tol +
                             func(point - 3 * tol)/ tol)
            elif accuracy ==3:
                deriv.append(-35 * func(point)/(12 * tol) +
                             26 * func(point - tol)/(3 * tol) +
                             -19 * func(point - 2 * tol)/(2 * tol) +
                             14 * func(point - 3 * tol)/ (tol * 3) +
                             -11* func(point - 4 * tol)/ (tol * 12))
            else:
                print "Wrong accuracy. Use 1, 2, or 3 for kind 'backward'"
    else:
        print "Not going to work. Check parameters"
    return deriv

# Problem 2
def the_func(x):
    return sp.sin(x)

tol1 = numerical_derivative(the_func,1, 'centered', 2, 1, tol = 1e-5)
tol2 = numerical_derivative(the_func,1, 'centered', 2, 1, tol = 1e-9)
true_dev = sp.cos(1)

error1 = tol1 - true_dev
error2 = tol2 - true_dev

error_convergence = error1/error2
