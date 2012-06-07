"""
Created May 15, 2012

Author: Spencer Lyon

Lab 27
"""
import numpy as np
import scipy as sp
import scipy.optimize as optim
import matplotlib.pyplot as plt
from numpy.matlib import repmat
import math
from scipy import random

# Problem 1
def basin(coefficient_list):
    """
    This function uses root solving methods to produce a plot of the basin's
    of attraction for a given quadratic polynomial.

    Inputs:
        coefficient_list: A list with three elements. The first is the
                         coefficient is the number in front of the x**2 term,
                         the second is the coefficent for the x term, and the
                         third is the constant term.

    Outputs:
        roots: These are the roots of the polynomial.
        plot: A plot is automatically generated that shows the quadratic
              polynomial along with the basin of attraction for each of the two
              roots.
    """

    func = np.poly1d(coefficient_list)
    roots = func.r
    if np.iscomplex(roots).any():
        print "Sorry there are complex roots. Try different coefficents"
        return []

    tries = np.linspace(min(roots) - 2, max(roots) + 2, 100)
    ans_root1 = []
    ans_root2 = []

    for i in range(0, tries.size):
        if (optim.newton(func, tries[i]) -.001 < roots[0]
            < optim.newton(func, tries[i]) +.001):
            ans_root1.append(0)
        else:
            ans_root2.append(0)

    ans_root1 = np.array(ans_root1)
    ans_root2 = np.array(ans_root2)

    plt.figure()
    plt.plot(tries,func(tries),'g')
    plt.plot(tries[0:ans_root1.size], ans_root1, 'bo',
             label = str('x -> ' + str(roots[0])))

    plt.plot(tries[ans_root1.size:], ans_root2, 'ro',
             label = str('x -> ' + str(roots[1])))

    plt.legend(loc = 0)

    plt.show()

    return roots

def runProblem1():
    x = random.randint(-5,5,(3))
    basin(x)

# Problem 2
def cubic_basin(coefficent_list, root_boot=False):
    """
    This function takes a polynomial of degree n and plots it along with the
    basin of attractions for each root.

    Inputs:
        coefficent_list: This is a list of coefficents. Starting with the last
                         element of the list, the number there will be the
                         coefficient first for the constant term, then the x
                         term, then x**2 and so on. The number of elements in
                         the list is one greater than the degree of the
                         polynomial.
        root_bool: This is a boolean value with default value set to False. If
                   it is changed to True then instead of each element of the
                   list corresponding to a coefficient of x, it will instead
                   be a root of the polynomial. In this case the number of
                   elements in the list is equal to the degree of the polynomial

    Outputs:
        roots: This is just a list of the roots for your polynomial.
        plot: A plot is automatically generated that shows the quadratic
              polynomial along with the basin of attraction for each of the two
              roots.
    """

    func = np.poly1d(coefficent_list)
    roots = func.r
    roots = np.sort(roots)
    if np.iscomplex(roots).any():
        print "Sorry there are complex roots. Try different coefficents"
        return []

    tries = np.linspace(min(roots) - 2, max(roots) + 2, 30)
    ans_root1 = []
    ans_root2 = []
    ans_root3 = []

    for i in range(0, tries.size):
        root = optim.newton(func, tries[i])
        if root -.0001 < roots[0] < root +.0001:
            ans_root1.append(0)
        elif root -.0001 < roots[1] < root +.0001:
            ans_root2.append(0)
        else:
            ans_root3.append(0)

    ans_root1 = np.array(ans_root1)
    ans_root2 = np.array(ans_root2)
    ans_root3 = np.array(ans_root3)

    plt.figure()
    plt.plot(tries,func(tries),'g')
    plt.plot(tries[0:ans_root1.size], ans_root1, 'bo',
             label = str('x -> ' + str(roots[0])))

    plt.plot(tries[ans_root1.size:ans_root1.size + ans_root2.size], ans_root2,
             'ro', label = str('x -> ' + str(roots[1])))

    plt.plot(tries[ans_root1.size + ans_root2.size:], ans_root3, 'yo',
             label = str('x -> ' + str(roots[2])))

    plt.legend(loc = 0)

    plt.show()

    return roots



def runProblem2():
    x = [1,0,-1,0]
    cubic_basin(x)

# Problem 3
# ----------------From Ernetso P. Adorio in the Phillipines--------------------#
from cmath import *
def zermuller(f, xinit, ztol= 1.0e-5, ftol=1.0e-5, maxiter=1000, wantreal=False, nroots=1):
    nmaxiter = 0
    retflag  = 0
    roots = []
    for j in range(nroots):
        x1  = xinit
        x0  = x1 - 1.0
        x2  = x1 + 1.0

        f0, undeflate  = deflate(f, x0, j, roots)
        f1, undeflate  = deflate(f, x1, j, roots)
        f2, undeflate  = deflate(f, x2, j, roots)

        h21 = x2 - x1
        h10 = x1 - x0
        f21 = (f2 - f1) / h21
        f10 = (f1 - f0) / h10

        for i in range(maxiter):
            f210 = (f21 - f10) / (h21+h10)
            b    = f21 + h21 * f210
            t    = b*b- 4.0 * f2 * f210

            if (wantreal) :     	# force real roots ? #
               if (real(t) < 0.0):
                   t = 0.0
               else :
                   t =  real(t)

            Q = sqrt(t)
            D = b + Q
            E = b - Q

            if (abs(D) < abs(E)) :
                D = E


            if (abs(D) <= ztol) :      # D is nearly zero ? #
                xm = 2 * x2 - x1
                hm = xm - x2
            else :
                hm = -2.0 * f2 / D
                xm = x2 + hm


            # compute deflated value of function at xm.  #
            fm, undeflate = deflate(f, xm, j, roots)


            # Divergence control #
            absfm = abs(fm)
            absf2 = 100. * abs(f2)
            # Note: Originally this was a while() block but it
            #       causes eternal cycling for some polynomials.
            #       Hence, adjustment is only done once in our version.
            if (absf2 > ztol and absfm >= absf2) :
                hm    = hm * 0.5
                xm    = x2 + hm
                fm    = f(xm)
                absfm = abs(fm)


            # function or root tolerance using original function
            if (abs(undeflate) <= ftol or abs(hm) <= ztol) :
                if (i > nmaxiter) :
                    nmaxiter = i
                    retflag = 0
                    break

            # Update the variables #
            x0  = x1
            x1  = x2
            x2  = xm

            f0  = f1
            f1  = f2
            f2  = fm

            h10 = h21
            h21 = hm
            f10 = f21
            f21 = (f2 - f1) / h21


        if (i > maxiter) :
            nmaxiter = i
            retflag  = 2
            break

        xinit = xm
        roots.append(xinit)

        # initial estimate should be far enough from latest root.
        xinit    = xinit + 0.85

    maxiter = nmaxiter
    return roots


def deflate(f,z, kroots, roots):
    """
    Arguments
      f                 Input: complex<double> function whose root is desired
      z                 Input: test root
      kroots            Input: number of roots found so far
      roots             Input/Output: saved array of roots

    Return value
      Deflated value of f at z.

    Description
      This routine is local to zermuller.
      Basically, it divides the complex<double> function f by the product
           		(z - root[0]) ... (z - root[kroots - 1]).
      where root[0] is the first root, root[1] is the second root, ... etc.
    """
    undeflate = t = f(z)
    nroots = len(roots)
    for i in range(nroots):
        denom = z - roots[i]
        while (abs(denom) < 1e-8):# avoid division by a small number #
            denom += 1.0e-8
        t = t / denom
    return t, undeflate

# -----------------------------------------------------------------------------#

def complex_quad_basin(coefficient_list):
    """
    This function uses root solving methods to produce a plot of the basin's
    of attraction for a given quadratic polynomial.

    Inputs:
        coefficient_list: A list with three elements. The first is the
                         coefficient is the number in front of the x**2 term,
                         the second is the coefficent for the x term, and the
                         third is the constant term.

    Outputs:
        roots: These are the roots of the polynomial.
        plot: A plot is automatically generated that shows a grid
    """

    func = np.poly1d(coefficient_list)
    roots = func.r
    roots  = np.sort(roots)


    if np.iscomplex(roots).any():

        xs1 = []
        xs2 = []
        xs3 = []

        ys1 = []
        ys2 = []
        ys3 = []

        tries_real = np.linspace(min(np.real(roots)) - .5,
                                 max(np.real(roots)) + .5, 100)

        tries_complex = np.linspace(min(np.imag(roots)) - .5,
                                 max(np.imag(roots)) + .5, 100)
        for i in tries_real:
            for k in tries_complex:
                root = zermuller(func, i+ k*1.j,maxiter = 500)
                if abs(root - roots[0]) < .0001:
                    xs1.append(i)
                    ys1.append(k)
                elif abs(root - roots[1])< .0001:
                    xs2.append(i)
                    ys2.append(k)
                else:
                    xs3.append(i)
                    ys3.append(k)

        plt.plot(xs1,ys1,'o', label = str('x -> ' + str(roots[0])))
        plt.plot(xs2,ys2,'go', label = str('x -> ' + str(roots[1])))
        if roots.size == 3:
            plt.plot(xs3,ys3,'yo', label = str('x -> ' + str(roots[2])))

        plt.legend(loc = 0)
        plt.show()



    else:
        func = np.poly1d(coefficient_list)
        roots = func.r
        roots = np.sort(roots)
        if np.iscomplex(roots).any():
            print "Sorry there are complex roots. Try different coefficents"
            return []

        tries = np.linspace(min(roots) - 2, max(roots) + 2, 30)
        ans_root1 = []
        ans_root2 = []
        ans_root3 = []

        for i in range(0, tries.size):
            root = optim.newton(func, tries[i])
            if root -.0001 < roots[0] < root +.0001:
                ans_root1.append(0)
            elif root -.0001 < roots[1] < root +.0001:
                ans_root2.append(0)
            else:
                ans_root3.append(0)

        ans_root1 = np.array(ans_root1)
        ans_root2 = np.array(ans_root2)
        ans_root3 = np.array(ans_root3)

        plt.figure()
        plt.plot(tries,func(tries),'g')
        plt.plot(tries[0:ans_root1.size], ans_root1, 'bo',
                 label = str('x -> ' + str(roots[0])))

        plt.plot(tries[ans_root1.size:ans_root1.size + ans_root2.size], ans_root2,
                 'ro', label = str('x -> ' + str(roots[1])))

        if roots.size == 3:
            plt.plot(tries[ans_root1.size + ans_root2.size:], ans_root3, 'yo',
                     label = str('x -> ' + str(roots[2])))

        plt.legend(loc = 0)

        plt.show()

    return roots

def runProblem3():
    x = [1,0,0,-1]
    complex_quad_basin(x)

# Problem 4

def julia_sets(complex_number, n, maxIter=30, a=-1, b=1):
    """
    This function computes the Julia set of the comlex number passed in
    according to the function f(x) = x**2  + c.

    Inputs:
	complex_number: The comlpex number for which you would like to find
			the Julia set.
	n: The size of the square (n x n) matrix that will contain plot
	   information.
	maxIter: The maximum number of iterations to pass x though f(x).
	a: The lower bound for both the real and complex axes.
	b: The upper bound for both the real and complex axes.

    Outputs:
	plot: There is no output that can be stored to a variable. When this
	      function is called a plot is automatically generated.
    """

    if n % 2 != 0:
	print 'Value Error: We need n to be an even number'
	return

    X = sp.zeros((n,n), dtype = 'complex64')
    Xj = sp.zeros((n,n), dtype = 'complex64')
    x = sp.linspace(a,b,n)
    xj = x * 1j

    X = repmat(x, n, 1)
    Xj = repmat(xj, n, 1)

    start_grid =  X + Xj.T
    answer_grid = sp.zeros((n,n), dtype = 'complex64')

    func = lambda x: x**2 + complex_number

    for i in range(n):
	for j in range(n):
	    x = start_grid[i,j]
	    for m in range(maxIter):
		x = func(x)
		m += 1
	    answer_grid[i,j] = x

    for i in range(n):
	for j in range(n):
	    if math.isnan(answer_grid[i,j]):
		answer_grid[i,j] = 0

    E = sp.zeros((n,n), dtype = 'complex64')
    for i in range(n):
	for j in range(n):
	    E[i,j] = sp.exp(-abs(answer_grid[i,j]))

    E = -1* sp.array(E, dtype = float)
    plt.imshow(E)

def runProblem4():
    cnum = 0.4 + 0.3j
    n = 30
    julia_sets(cnum,n)
