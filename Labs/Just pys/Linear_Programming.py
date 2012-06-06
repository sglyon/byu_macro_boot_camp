"""
Created May 22, 2012

Author: Spencer Lyon
"""

from cvxopt import matrix, solvers
import numpy as np
import numpy.matlib as matlib
from numpy import linalg as la
import random
import math
import matplotlib.pyplot as plt


def lineData(n,noise=1.0):
    """
    Produces n noisy points on a randomly generated line in R^2.
    """

    xs = [-10 + 20.0*i/n for i in range(n)]
    m = random.normalvariate(0,10)
    b = random.normalvariate(0,2)
    ys = [m*xs[i]+b+5*random.normalvariate(0,noise) for i in range(n)]

    return xs,ys

def ellipseData(n,noise=1.0):
    """
    Produces n noisy points on a randomly generated ellipse in R^2.
    """

    theta = [2*math.pi*i/float(n) for i in range(n)]
    r = 8*random.random()

    D = matrix([[r*math.cos(theta[i]) for i in range(n)], [r*math.sin(theta[i]) for i in range(n)]])
    T = matrix([[8*random.normalvariate(0,1) for i in range(2)] for j in range(2)])
    D = D*T;

    xshift = random.normalvariate(0,4)
    yshift = random.normalvariate(0,4)

    xs = [D[i,0] + xshift + random.normalvariate(0,noise) for i in range(n)]
    ys = [D[i,1] + yshift + random.normalvariate(0,noise) for i in range(n)]
    return xs,ys

def plotEllipse(beta,n):
    """
    Generates n points lying on the ellipse defined by beta = [a, b, c, d, e]
    """

    # get the parameters from beta
    a,b,c,d,e = beta
    # convert to the quadratic form model
    A = matrix([[float(a), float(b)/float(2.0)],[float(b)/float(2.0), float(c)]])

    # diagonalize the model
    D,Q = np.linalg.eig(A)
    Q = matrix(Q)
    lambda1 = D[0]
    lambda2 = D[1]

    if (lambda1<0) or (lambda2<0):
        print 'Degenerate Ellipse'
        return None,None

    # apply change of variables to linear terms
    B = matrix([[float(d)],[float(e)]])*Q
    d = B[0,0]
    e = B[0,1]

    # compute the center by completing the square
    x0 = -d/(2.0*lambda1)
    y0 = -e/(2.0*lambda2)

    # shift the constant term from completing the square
    f = 1 + d**2/(4.0*lambda1) + e**2/(4.0*lambda2)

    # get the radii of the ellipse (set y=0 to find rx, etc)
    rx = math.sqrt(f/lambda1)
    ry = math.sqrt(f/lambda2)

    # generate the points on the standard form ellipse
    theta = [i*2*math.pi/float(n) for i in range(n)]
    D = matrix([[rx*math.cos(theta[i])+x0, ry*math.sin(theta[i])+y0] for i in range(n)])

    # apply Q to rotate to correct orientation, then shift to center
    E = Q*D
    x = [E[0,i] for i in range(n)]
    y = [E[1,i] for i in range(n)]

    return x,y


def problem_1():
    """
    This function provides the solution for problem 1 in the file:
    'lab3 - linaer-programming.pdf'

    Inputs:
        None

    Returns:
        A dictionary containing the solution and a lot of other info as defined
        in cvxopt.solvers.lp
    """
    c = matrix([-1.0, -2.0, -3.0, -4.0, -5.0])
    G = matrix([[5.0, 4.0, 3.0, 2.0, 1.0,],
                [1.0, -2.0, 3.0, -4.0, 5.0],
                [-1.0, 2.0, -3.0, 4.0, -5.0],
                [-1.0, 0, 0 ,0 ,0],
                [0, -1.0, 0, 0, 0],
                [0, 0, -1.0, 0, 0],
                [0, 0, 0, -1.0, 0],
                [0, 0, 0, 0, -1.0]])
    G = G.T
    h = matrix([30.0, 20.0, 10.0, 0., 0., 0., 0., 0.])
    solution = solvers.lp(c, G, h)
    return solution

def problem_2():
    """
    This function provides the solution for problem 2 in the file:
    'lab3 - linaer-programming.pdf'

    Inputs:
        None

    Returns:
        A dictionary containing the solution and a lot of other info as defined
        in cvxopt.solvers.lp
    """
    c = matrix([-6., -8., -5., -9.])
    G = matrix([[2., 1., 1., 3.],
                [1., 3., 1., 2.],
                [-1., 0., 0., 0.],
                [0., -1.0, 0., 0.],
                [0., 0., -1.0, 0.],
                [0., 0., 0., -1.0]])
    G = G.T
    h = matrix([5., 3., 0., 0., 0., 0.])
    A = matrix([1., 1., 1.,0.])
    A = A.T
    b = matrix([1.])
    solution = solvers.lp(c, G, h, A, b)
    return solution

def problem_3():
    """
    This function provides the solution for problem 3 in the file:
    'lab3 - linaer-programming.pdf'

    Inputs:
        None

    Returns:
        A dictionary containing the solution and a lot of other info as defined
        in cvxopt.solvers.lp
    """
    c = matrix([-2.0, 1.0 ,-9.0])
    G = matrix([[-1.0, -1.0, 3.0],
               [1.0, 0.0, 2.0,],
               [-1.0, 0.0, 0.0],
               [0.0, -1.0, 0.0],
               [0.0, 0.0, -1.0]]).T
    h = matrix([3.0, 2.0, 0.0, 0.0, 0.0])
    solution = solvers.lp(c, G, h)
    x1 = np.array(1 - solution['x'][0] - solution['x'][1])
    sol = np.array(solution['x'])
    return [np.insert(sol,0 , x1), solution]

def problem_4():
    """
    This function provides the solution for problem 4 in the file:
    'lab3 - linaer-programming.pdf'

    Inputs:
        None

    Returns:
        A dictionary containing the solution and a lot of other info as defined
        in cvxopt.solvers.lp
    """
    c = matrix([-5.0,-4.0])
    G = matrix([[1.0, 1.0],
                [ -2.0, -2.0],
                [-1.0, 0.0],
                [0.0, -1.0]]).T
    h = matrix([2.0, -9.0, 0.0, 0.0])
    solution = solvers.lp(c, G, h)
    print " THE 'status' FOR THIS PROBLEM IS...", solution['status']
    return solution

def problem_5():
    """
    This function provides the solution for problem 5 in the file:
    'lab3 - linaer-programming.pdf'

    Inputs:
        None

    Returns:
        A dictionary containing the solution and a lot of other info as defined
        in cvxopt.solvers.lp
    """
    c = matrix([-1.0, 4.0])
    G = matrix([[-2.0, 1.0],
                [-1.0, -2.0],
                [-1.0, 0.0],
                [0.0, -1.0]]).T
    h = matrix([-1.0, -2.0, 0.0, 0.0])
    solution = solvers.lp(c, G, h)
    print " THE 'status' FOR THIS PROBLEM IS...", solution['status']
    return solution

def problem_6():
    """
    This function provides the solution for problem 6 in the file:
    'lab3 - linaer-programming.pdf'

    Inputs:
        None

    Returns:
        A dictionary containing the solution and a lot of other info as defined
        in cvxopt.solvers.lp
    """
    # Building parts for c vector
    fcost = 1.0
    ccost = .75
    mcost = 4.25
    ncost = 2.75
    crcost = 2.1
    PrA = np.array([(2.5 *.8 - fcost), (2.5 *.05 - ccost), (2.5 *.02 - mcost),
                    (2.5 *.05 - ncost), (2.5 *.08 - crcost)])
    PrB = np.array([(1.8*.93 - fcost), (1.8*.05 - ccost), 0, 0,
                    (1.8*.02 - crcost)])
    PrC = np.array([1.3 * .99 -fcost, 1.3 * .01 - ccost, 0, 0, 0], dtype = float)
    cNp = -1 * np.hstack((PrA, PrB, PrC))
    c = matrix(cNp)

    # Building parts for G matrix.
    rowA = np.hstack((np.array([.8, .05, .02, .05, .08]), np.zeros(10)))
    rowB = np.hstack((np.zeros(5), np.array([0.93, 0.05, 0, 0, 0.02]),
                      np.zeros(5)))
    rowC = np.hstack((np.zeros(10), np.array([0.99, 0.01, 0, 0, 0])))
    eyeBlock = np.hstack((np.eye(5), np.eye(5), np.eye(5)))
    timeRow = 8 *rowA + 6*rowB + 2* rowC
    bigEye =  - np.eye(15)
    GNp = np.vstack((rowA, rowB, rowC, eyeBlock, timeRow, bigEye))
    G = matrix(GNp)

    # h vector
    no_neg = np.zeros(15)
    constraints = np.array([5000.0, 15600.0, 8500.0,
                15000.0, 10000.0, 2000.0, 3500.0, 10000.0, 120000.0])
    hNp = np.hstack((constraints, no_neg))
    h = matrix(hNp)

    solution = solvers.lp(c, G, h)
    x = solution['x']
    profit = np.dot( - cNp, x)
    print 'The optimal profit is ', profit
    return solution

def problem_9(X, y):
    """
    This function implements the minimization of ||X*beta - y|| for the infinity
    norm.

    Inputs:
        X: The matrix X that defines the characterizing equations for the model.
        y: The vector so that X*beta = y

    Returns:
        beta: A NumPy array representing the vector that minimizes the objective
        function.

    Notes:
        This function uses the cvxopt.solvers.lp routine to get the answer.
    """
    X = np.mat(X, dtype = float)
    y = np.array(y, dtype = float)

    m, n  = X.shape
    onesT = np.ones((m, 1))
    GNp = np.vstack((np.hstack((X, -onesT)), np.hstack((-X, -onesT))))
    hNp = np.hstack((y, -y))
    cNp = np.hstack((np.zeros(n), np.ones(1)))

    G = matrix(GNp)
    h = matrix(hNp)
    c = matrix(cNp)

    solution  = solvers.lp(c,G,h)
    return np.array(solution['x'])[:n]

def problem_10(X, y):
    """
    This function implements the minimization of ||X*beta - y|| for the one
    norm.

    Inputs:
        X: The matrix X that defines the characterizing equations for the model.
        y: The vector so that X*beta = y

    Returns:
        beta: A NumPy array representing the vector that minimizes the objective
        function.

    Notes:
        This function uses the cvxopt.solvers.lp routine to get the answer.
    """
    X = np.mat(X, dtype = float)
    y = np.array(y, dtype = float)

    m,n = X.shape
    B = np.eye(m)
    G = np.vstack((np.hstack((X, -B)), np.hstack((-X, -B))))
    c = np.hstack((np.zeros(n), np.ones(m)))
    h = np.hstack((y, -y))

    c = matrix(c)
    h = matrix(h)
    G = matrix(G)
    solution = solvers.lp(c, G, h)
    return np.array(solution['x'])[:n]

def problem_11(X, y):
    """
    This function implements the minimization of ||X*beta - y|| for the two
    norm.

    Inputs:
        X: The matrix X that defines the characterizing equations for the model.
        y: The vector so that X*beta = y

    Returns:
        beta: A NumPy array representing the vector that minimizes the objective
        function.

    Notes:
        This function uses the numpy.la.solve routine to get the answer.
    """
    ans = la.lstsq(X,y)
    return ans[0]

def problem_12(x, y):
    """
    This function takes in two vectors (x, y) and returns the slope (m) and
    intercept (b) that minimizes ||x* (m+b) - y||  according to the
    1-norm, the infinity-norm, and the 2-norm.

    Inputs:
        x: A vector of x coordinates
        y: A vector of y coordinates

    Outputs:
        Beta1: This a numpy array with [m, b] that minimizes the objective
               function according to the 1-norm.

        Beta2: This a numpy array with [m, b] that minimizes the objective
               function according to the 2-norm.

        BetaInf: This a numpy array with [m, b] that minimizes the objective
               function according to the Infinity-norm.

    Notes:
        This function calls the functions problem_9, problem_10, and problem_11
        . That also means that cvxopt.solvers.lp and numpy.linalg.solve are
        needed.

        Also, to generate the x and y you can use the function lineData from
        at the top of this file.
    """
    xColumn = np.mat(x).T
    n = xColumn.size
    y = np.array(y)
    x = np.array(x)

    xMat = np.mat(np.hstack((xColumn, np.ones((n,1)))))

    BetaInf = problem_9(xMat, y)
    Beta1 = problem_10(xMat, y)
    Beta2 = problem_11(xMat, y)

    f1 = lambda xarg: Beta1[0] * xarg + Beta1[1]
    fInf = lambda xarg: BetaInf[0] * xarg + BetaInf[1]
    f2 = lambda xarg: Beta2[0] * xarg + Beta2[1]

    f1 = np.vectorize(f1)
    fInf = np.vectorize(fInf)
    f2 = np.vectorize(f2)


    plt.figure()
    plt.plot(x, y, 'o')
    plt.plot(x,f1(x), 'k', label = 'Using 1 Norm')
    plt.plot(x,fInf(x), 'r', label = 'Using Infinity Norm')
    plt.plot(x,f2(x), 'y', label  = 'Using 2 Norm')
    plt.legend(loc = 0)
    plt.show()



    return [Beta1, Beta2, BetaInf]

def problem_13(x, y):
    """
    This function takes in two vectors (x, y) and returns the coefficients
    a, b, c, d, e from the general equation for an ellipse (a * x**2 + b * x*y +
    c * y**2 + d * x + e * y =1). This will be computed by minimizing both the
    one and the two norm of the associated matrix equation.

    Inputs:
        x: A vector of x coordinates.
        y: A vector of y coordinates.

    Outputs:
        Beta1: This a numpy array with [a, b, c, d, e] that minimizes the
               objective function according to the 1-norm.

        Beta2: This a numpy array with [a, b, c, d, e] that minimizes the
               objective function according to the 2-norm.

    Notes:
        This function calls the functions problem_10 and problem_11. That also
        means that cvxopt.solvers.lp and numpy.linalg.solve are needed.

        Also, to generate the x and y you can use the function lineData from
        at the top of this file.
    """
    yArray = np.array(y)
    xArray = np.array(x)
    n = xArray.size

    Column_1 = np.mat(xArray**2).T
    Column_2 = np.mat(xArray * yArray).T
    Column_3 = np.mat(yArray**2).T
    Column_4 = np.mat(x).T
    Column_5 = np.mat(y).T



    xMat = np.mat(np.hstack((Column_1, Column_2, Column_3, Column_4, Column_5)))

    b = np.ones(n)

    Beta1 = problem_10(xMat, b)
    Beta2 = problem_11(xMat, b)

    xbeta1, ybeta1 = plotEllipse(Beta1, n)
    xbeta2, ybeta2 = plotEllipse(Beta2, n)

    plt.figure()
    plt.plot(xbeta1, ybeta1, label = 'Using 1-Norm')
    plt.plot(xbeta1, ybeta1, label = 'Using 2-Norm')
    plt.legend(loc = 0)
    plt.show()

    return [Beta1, Beta2]
