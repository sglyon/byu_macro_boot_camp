"""
Created May 31, 2012

Author: Spencer Lyon
"""
import numpy as np
import scipy as sp
import scipy.optimize as opt
import scipy.linalg as la

# For problem 1 the answers to the 1 variable optimization problem are,
# local max at y = 1/28*(13 - 533**(1/2)) ~= -.36 and local min at
# y = 1/28*(13 + 533**(1/2)) ~=1.29

def problem_1():
    """
    This problem uses scipy.optimize.fmin_slsqp to solve the 1 variable
    optimization problem for the local min and max noted above.

    Inputs:
        None

    Outputs:
        y: List of y values for the local min
        f: The function evaluated at the point in y.
    """
    func_min = lambda y: -28*y**3 + 39*y**2 + 39*y - 117
    func_max = lambda y: 28*y**3 - 39*y**2 - 39*y + 117

    ymin = opt.fmin(func_min, -.3)
    ymax = opt.fmin(func_max, 1.0)

    val_min = func_min(ymin)
    vam_max = func_min(ymax)

    y = [ymin, ymax]
    val = [val_min, vam_max]

    return y, val

def problem_2_1():
    """
    This problem uses scipy.optimize.fmin_slsqp to solve the constrained
    optimization problem in part one of problem 2.
    """
    func = lambda x: (x[0]**2 + 2 * x[0] * x[1] + 3 * x[1]**2 + 4 * x[0] +
                      5 * x[1] + 6 * x[2])
    f_eqcons = lambda x: sp.array([x[0] + 2 * x[1] - 3,
                                  4 * x[0] + 5*x[2] - 6])
    x,fx,its,imode,smode = opt.fmin_slsqp(func,
                                          sp.array([0,-1,0]),
                                          [],
                                          f_eqcons,
                                          full_output = 1)
    return [x, fx, smode ]

def problem_2_2():
    """
    This problem uses scipy.optimize.fmin_slsqp to solve the constrained
    optimization problem in part two of problem 2.
    """
    func = lambda x: (4 * x[0] + x[1]**2)
    f_eqcons = lambda x: sp.array([x[0]**2 + x[1]**2 - 9])
    x,fx,its,imode,smode = opt.fmin_slsqp(func,
                                          sp.array([1,1]),
                                          [],
                                          f_eqcons,
                                          full_output = 1)

    return [x, fx, smode ]

def problem_2_3():
    """
    This problem uses scipy.optimize.fmin_slsqp to solve the constrained
    optimization problem in part three of problem 2.
    """
    func = lambda x: (x[0] * x[1])
    f_eqcons = lambda x: sp.array([x[0]**2 + 4 * x[1]**2 - 1])
    x,fx,its,imode,smode = opt.fmin_slsqp(func,
                                          sp.array([1,1]),
                                          [],
                                          f_eqcons,
                                          full_output = 1)

    return [x, fx, smode ]

def qpe((P,q,r), (C,d), want_val=1):
    """
    This function solves the equality constrained quadratic program (QPE):

        minimize 1/2 x.T * P * x + a.T * x + r, P >=0
        s. t. : Cx = d

    This can be converted into the following matrix equation:
        [[P, C.T], * [[x], = [[-q],
         [C, 0]]      [v]]    [d]]

    Inputs:
        P, q, r, C, d: The matricies outlined in the obejctive function
                       and constraints above.

    Outputs:
        x: The x that solves the matrix equation above.
        v: The v that solves the matrix equation above.
        val (optional): The value of the objective funciton at the optimal x.
    """
    P, q, r, C, d = np.mat(P), np.mat(q), np.mat(r), np.mat(C), np.mat(d)
    vsize, xsize = C.shape
    big_A = np.vstack((np.hstack((P, C.T)),
                       np.hstack((C, np.zeros((vsize, vsize))))))
    big_b = np.vstack((-q.T, d.T))

    big_ans = la.solve(big_A, big_b)

    x = big_ans[0:xsize]
    v = big_ans[xsize:]

    if want_val ==1 :
        val = 1/2 * x.T * P * x + q * x + r
        return x, v, val

    else:
        return x, v

def problem_4():
    """
    This function solves the equality constrained least squared optimization
    problem by converting it into a QPE problem and using the function qpe
    defined above.

    We start with this problem:
        minimize ||Ax - b||_2
        st:        Cx = d

    This can be converted into QPE form by expanding the two norm to get:
        ||Ax-b|| = X.T * A.T * A * x - 2 b.T * A + b.T * b

    Comparing this to the general form we see that we define the matricies as
    follows:

        P = 2 A.T * A
        q.T = -2b.T * A
        r = b.T * b
        C = C
        d = d

    Inputs:
        None

    Outputs:
        x: The x that solves the matrix equation above.
        v: The v that solves the matrix equation above.
        val: The value of the objective funciton at the optimal x.

    Notes:
        This function uses the qpe function from above.
    """

    A = np.array([[1, 1, 1],
                  [1, 3, 1],
                  [1, -1, 1],
                  [1, 1, 1]])
    b = np.array([1,2,3,4])

    C = np.array([[1, 1, 1], [1, 1, -1]])
    d = np.array([7,4])

    P = 2 * np.dot(A.T, A)
    q = -2 * np.dot(b.T, A).T
    r = np.dot(b.T, b)

    func = lambda x: x.T * A.T * A * x - 2 * b.T * A * x + b.T * b


    x, v, val = qpe((P,q,r), (C,d))

    return x, v, val

def problem_5():
    """
    This problem transforms using Newton's method for root fiding into a
    QPE problem. The form of the problem is just the following linear system:

        [[ H(f(x0)), C.T ]  * [[delta_x],   =  [-J(f(x_0))
         [    C    ,  0 ]]     [   v   ]]           0     ]

    For this specific problem our objective function and constraints are:

        minimize: sum(x_j * log(x_j), {x,1,n})
        st      : Cx = d

    Inputs:
        None

    Outputs:
        x: The x that solves the matrix equation above.
        v: The v that solves the matrix equation above.
        val: The value of the objective funciton at the optimal x.

    Notes:
        This function uses the QPE function from above.
        Also we will be generating a random matrix C, a random x0 and using them
            to generate a random d vector. This ensures that we are starting at
            a feasible point.

    """
    def jaco(x):
        """
        This is a simple implementation of our jabobian matrix. In this case
        the Jacobian is equal to the gradient and is a one dimensional vector.

        From the obejctive funciton we see that the jacobian is just the
        partial of each x. This gives us:
            J[i] = log(x[i]) + x[i]/x[i] = log(x[i]) + 1

        Inputs:
            x: The current value of the vector x

        Outputs:
            jac: The Jacobian for the given current x.
        """
        x = np.array(x)
        return np.array(np.log(x) + 1)

    def hess(x):
        """
        This is a simpe implementation of our Hessian matrix. Because the
        objective function contains no cross product terms, the Hessian becomes
        a diagonal matrix with the second order partial along the diagonal.

        Starting at the first partial as found in the function jaco we take one
        more derivative to get:
            H[i,i] = 1/x[i]

        Inputs:
            x: The current value of the vector x

        Outputs:
            jac: The Hessian Matrix for the given current x.

        """
        x = np.array(x)
        return np.diag(1.0/x)


    n = 40
    p = 10

    x0 = sp.rand(n)

    C = sp.rand(p, n)
    d = np.dot(C,x0)

    P = hess(x0)
    q = jaco(x0)
    delta_x = qpe((P, q, 0), (C, np.zeros(p)), 0)[0].flatten()
    xnew = x0 + delta_x

    count = 1
    while la.norm(delta_x) > 1e-7:
        count += 1
        P = hess(xnew)
        q = jaco(xnew)
        delta_x = qpe((P, q, 0), (C, np.zeros(p)), 0)[0].flatten()
        xnew += delta_x

    return xnew
