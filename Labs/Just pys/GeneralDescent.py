"""
Created May 17, 2012

Author: Spencer Lyon

Optimization.py

Some changes
"""
import scipy as sp
import numpy as np
import scipy.linalg as la

# Problem 1
def steepest_descent(A, b, x0, tol=1e-8):
    """
    Uses the steepest descent method to find the x that satisfies Ax = b.

    Inputs:
        A: An m x n NumPy array
        b: An m x 1 NumPy array
        x0: An n x 1 NumPy array that represents the initial guess at a
            solution.
        tol (optional): The tolerance level for convergence. This is compared
                        against the norm(x_n+1 - x_n) each iteration.

    Outputs:
        x: The x that satisfies the equation.
    """
    A = sp.mat(A)
    b = sp.reshape(sp.mat(b),(b.size,1))


    def grad(A, b, x):
        """
        Find the gradient of ||Ax - b||
        Inputs:
            A: An m x n NumPy matrix.
            b: An m x 1 NumPy matrix.
            x: An n x a NumPy matrix.

        Outputs:
            grad: A NumPy matrix representing the gradient of ||Ax - b||
        """
        return np.mat(2  * A.T*(A*x - b))

    def solve_alpha_k(A, b, x):
        """
        Solves for alpha in the steepest descent algorithm
        x_n+1 = x_n - alpha * grad(x_n)

        Inputs:
            A: An m x n NumPy array
            b: An m x 1 NumPy array
            x: The x value where you want alpha to be defined for.

        Outputs:
            alpha: The alpha satisfying the algorithm above.
        """

        gradient = grad(A, b, x)
        return np.array(
            (gradient.T * gradient)/(2 * gradient.T * A.T * A * gradient))[0]



    xold = sp.reshape(sp.mat(x0),(x0.size,1))
    xnew = xold - grad(A, b, xold) * solve_alpha_k(A,b,xold)

    while la.norm(xold - xnew) > tol:
        xold = xnew
        xnew = xold - grad(A, b, xold) * solve_alpha_k(A,b,xold)

    return xnew

#sp.set_printoptions(suppress=True, precision = 5)
#run GeneralDescent.py
#A = sp.rand(8,5)
#b = sp.rand(8)
#x = sp.ones(5)
#my_ans_steepest = steepest_descent(A, b, x)
#my_ans_conj = conjugate_gradient(A, b, x)
#least_squares_ans = sp.mat(la.lstsq(A,b)[0]).T
#my_ans_steepest - least_squares_ans


# Problem 2
def conjugate_gradient(A, b, x0, tol=1e-8):
    """
    Uses the conjugate gradient method to find the x that satisfies Ax = b.

    Inputs:
        A: An m x n NumPy array
        b: An m x 1 NumPy array
        x0: An n x 1 NumPy array that represents the initial guess at a
            solution.
        tol (optional): The tolerance level for convergence. This is compared
                        against the norm(x_n+1 - x_n) each iteration.

    Outputs:
        x: The x that satisfies the equation.
    """
    A = sp.mat(A)
    b = sp.reshape(sp.mat(b),(b.size,1))
    xold = sp.reshape(sp.mat(x0),(x0.size,1))


    def grad(A, b, x):
        """
        Find the gradient of ||Ax - b||
        Inputs:
            A: An m x n NumPy matrix.
            b: An m x 1 NumPy matrix.
            x: An n x a NumPy matrix.

        Outputs:
            grad: A NumPy matrix representing the gradient of ||Ax - b||
        """
        return np.mat(2  * A.T*(A*x - b))


    def beta_k(A, b, xnew, xold):
        """
        Returns the beta satisfying the folloiwng equation:
            (g_n+1.T * g_n+1) / (g_n.T * g_n)

        Inputs:
            A: An m x n NumPy matrix.
            b: An m x 1 NumPy matrix.
            xnew: The x generated in the most recent iteration
            xold: The x generated in the previous iteration.

        Outputs:
            beta: The beta satisfying the equation.

        Notes:
            This function calls the grad function.

        """
        gnew = grad(A, b, xnew)
        gold = grad(A, b, xold)
        return (gnew.T * gnew) / (gold.T * gold)

    def dvector_k(A, b, xnew, xold, d_old):
        """
        Returns the vector d satisfying the equation:
            d_n+1  = - grad_n+1 + beta_n * d_n

        Inputs:
            A: An m x n NumPy matrix.
            b: An m x 1 NumPy matrix.
            xnew: The x generated in the most recent iteration
            xold: The x generated in the previous iteration.
            d_old: The d vector generated in the previous iteration

        Outputs:
            d: The vector d satisfying the above equation.

        Notes: This function calls the grad function and the beta_k function.
        """
        gnew = grad(A, b, xnew)
        bet = beta_k(A, b, xnew, xold)
        return (- gnew) + d_old* bet


    def solve_alpha_k_cg(A, b, xnew, xold, d_old):
        """
        Solves for alpha in the conjugate gradient algorithm
        x_n+1 = x_n + alpha * d(x_n)

        Inputs:
            A: An m x n NumPy matrix.
            b: An m x 1 NumPy matrix.
            xnew: The x generated in the most recent iteration
            xold: The x generated in the previous iteration.
            d_old: The d vector generated in the previous iteration

        Outputs:
            alpha: The alpha satisfying the algorithm above.

        Notes:
            This function calls the grad function and the dvector_k function.
        """

        gradient = grad(A, b, xnew)
        dvec = dvector_k(A, b, xnew, xold, d_old)
        return float(
            -(gradient.T * dvec)/(2 * dvec.T * A.T * A * dvec))



    grad0 = grad(A, b, xold)# g0

    if grad0.all() == 0:
        return xold
    else:
        d_old = - grad0 # d0

    alpha0 = - (grad0.T * d_old)/ (d_old.T * A.T* A * d_old)
    xnew = xold + d_old * alpha0 # x1



    g_old = grad(A, b, xnew) # g1
    beta_old = beta_k(A, b, xnew, xold) # beta0
    d_new = dvector_k(A, b, xnew, xold, d_old) # d1

    count = 0
    while la.norm(xold - xnew) > tol:
        xold_copy = xold #x0 -> x1
        xold = xnew # x1 -> x2

        d_old = d_new # d1 -> d2

        alpha_new = solve_alpha_k_cg(A, b, xnew, xold_copy, d_old) # a1 -> a2
        xnew = xold + d_old * alpha_new # x2 -> x3
        g_new = grad(A, b, xnew) # g2 -> g3
        beta_new = beta_k(A, b, xnew, xold) # beta1 -> beta2
        d_new = dvector_k(A, b, xnew, xold, d_old) # d2 -> d3
        count += 1
    return xnew

def runProblem2(m,n):
    A = sp.rand(m,n)
    b = sp.rand(m)
    x = sp.ones(n)

    return [conjugate_gradient(A, b, x), la.lstsq(A,b)[0], conjugate_gradient(A, b ,x) - np.mat(la.lstsq(A, b)[0]).T]
