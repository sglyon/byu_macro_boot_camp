"""
Created May 23, 2012

Author: Spencer Lyon
"""

import numpy as np
import scipy as sp
sp.set_printoptions(linewidth = 140, precision = 3, suppress = True)

def destroy_column(A, m, n):
    """
    This function performs a pivot operation given a tableau A,
    a row m, and a column n.

    Inputs:
        A: The tableau for which we want to perform the pivot
           operation.
        m: The row around which the pivot operation should take place.
        n: The column where we are pivoting.

    Outputs:
        A: The new matrix A with one column completely destroyed.
    """
    A = np.mat(A)
    rows, cols = A.shape

    A[m,:] /= A[m,n]

    for row in range(0,rows):
        if A[row,n] != 0 and row != m:
            A[row,:] += -A[m,:]*A[row,n]

    return A


def gen_Tableau(c, A, b):
    """
    This function will generate a tableau for the program
    in this form: [[1,c,0],[0,A,b]]

    Inputs:
        c: The objective function from the linear program
        A: The inequality constraints. All inequalities must be
           less than or equal to
        b: The constraining values for the inequalities.

    Outputs:
        Tab: The tableau from c, A, b.
    """
    c = np.mat(c)
    A = np.mat(A)
    b = np.mat(b)

    # Getting number of basic and non-basic vars
    basic, non_basic = A.shape

    b_bool = b < 0

    # Checking to see if we have any negative constraints.
    # If we do we are going to multiply to corresponding rows of A by -1
    if b_bool.any():
        loc = b_bool*np.eye(b.size)
        ind = np.where(loc != 0)[1]
        for row in range(ind.size):
            A[ind[0,row],:] /= -1

    b = abs(b)



    pad_row, pad_col = A.shape

    # Standard first column
    col_1 = np.vstack((np.array(1), np.mat(np.zeros(pad_row)).T))

    # Standard main block
    block_main = np.vstack((c, A))

    # Standard last column
    col_end = np.vstack((np.array(0), b.T))

    # Setting up the block that for columns for slack variables.
    slack_I = np.eye(pad_row)

    # If we had negative constraint we also need to multiply rows of slack_I
    # by -1

    if b_bool.any():
        for row in range(ind.size):
            slack_I[ind[0,row], ind[0,row]] /= -1

    slack_block = np.vstack((np.zeros(pad_row), slack_I))

    # Just putting blocks together.
    Tab = np.hstack((col_1, block_main, slack_block, col_end))

    return Tab

def test_feas(c, A, b):
    rows, cols = A.shape
    Tab = gen_Tableau(c, A, b)
    new_col1  = np.vstack([np.array(1), np.zeros((rows+2,1))])
    new_col2 = np.vstack([np.zeros((1, cols+1)), Tab[:,:-1]])
    new_col3 = np.vstack([-1*np.ones((1,rows)), np.zeros((1,2)), np.eye(rows)])
    new_col4 = np.vstack([np.array(0), Tab[:,-1]])

    aux_Tab = np.hstack([new_col1, new_col2, new_col3, new_col4])

    return aux_Tab

def find_pivot(T):
    """
    This function is used to find a pivot row and column for the
    given tableau T.

    Inputs:
        T: The Tableau for which you need to find the next pivot

    Outputs:
        newRow: The new pivot row.
        newCol: The new pivot column.
    """

    Tab = T.copy()

    newObj = np.array(Tab[0,1:-1] >0).flatten()
    newCol = np.nonzero(newObj)[0][0] + 1

    if (np.array(Tab[1:, newCol]).flatten() > 0).any():

        # select the row in the pivot column
        quotients = np.array(Tab[1:,-1]/Tab[1:,newCol]).flatten()
        for i in range(quotients.size):
            if quotients[i] < 0:
                quotients[i] = np.inf
        newRow = np.argsort(quotients)[0] + 1

    else:
        raise ValueError('The problem is unbounded')

    return [newRow,newCol]


def simplex(c, A, b):
    """
    This function maximizes the objective function c.T*x subject to
    Ax <= b, and x >= 0.

    Inputs:
        c: An n x 1 NumPy array representing the coefficients of
           the objective function.
        A: A NumPy matrix representing inequality constraints.
        b: The binding constraint values.

    Outputs:
        x: The optimal x vector
        value: The value of the objective function at x.
        exit_flag: Takes on 1 of three values. 1.) Solved.
                   2.) Infeasible 3.) unbounded.

    TODO:
        1.) Have checks for feasibilty and boundedness
        2.) Fix logic error of just going down one row at at time,
            doesn't always work.
        3.) Pull out x, value, and exit_flag.
    """
    def find_pivot(T):
        """
        This function is used to find a pivot row and column for the
        given tableau T.

        Inputs:
            T: The Tableau for which you need to find the next pivot

        Outputs:
            newRow: The new pivot row.
            newCol: The new pivot column.
        """

        Tab = T.copy()

        newObj = np.array(Tab[0,1:-1] >0).flatten()
        newCol = np.nonzero(newObj)[0][0] + 1

        if (np.array(Tab[1:, newCol]).flatten() > 0).any():

            # select the row in the pivot column
            quotients = np.array(Tab[1:,-1]/Tab[1:,newCol]).flatten()
            for i in range(quotients.size):
                if quotients[i] < 0:
                    quotients[i] = np.inf
            newRow = np.argsort(quotients)[0] + 1

        else:
            raise ValueError('The problem is unbounded')

        return [newRow,newCol]

    c = np.mat(c)
    A = np.mat(A)
    b = np.mat(b)


    # Getting number of basic and non-basic vars
    basic, non_basic = A.shape


    Tab = gen_Tableau(c, A, b)

    A1 = Tab[1:,1:non_basic+1]

    for col in range(non_basic):
        if not (A1[:,col] > 0).any():
            print 'Unbounded. Exiting'
            return [None, None, 'Unbounded']

    newObj = np.array([True, False])

    while newObj.any():
        newRow, newCol = find_pivot(Tab)
        Tab = destroy_column(Tab, newRow, newCol)
        newObj = np.array(Tab[0,1:-1] >0).flatten()

    final_obj = np.array(Tab[0,1:-1]).flatten()
    x = np.empty(non_basic)
    w = np.empty(basic)

    for col in range(x.size):
        if final_obj[col] != 0:
            x[col] = 0
        else:
            temp_col = np.array(Tab[:,col+1]).flatten()
            temp_row = np.nonzero(temp_col)[0][0]
            x[col] =  Tab[temp_row,-1] * Tab[temp_row, col+1]

    for col in range(x.size, w.size +x.size):
        if final_obj[col] != 0:
            w[col - x.size] = 0
        else:
            temp_col = np.array(Tab[:,col+1]).flatten()
            temp_row = np.nonzero(temp_col)[0][0]
            w[col- x.size] =  Tab[temp_row,-1] * Tab[temp_row, col+1]

    value = -1*Tab[0,-1]

    if (x <0).any() or (w < 0).any():
        print 'Unfeasible. Exiting'
        return [None, None, "Unfeasible"]


    return [x, value, 'Success']

## Easy check
#A = np.mat([[-4,-2],[-2,3],[1.,0]])
#c = np.array([1,2.])
#b = np.array([-8,6,3.])

## Unbounded Check
#A = np.mat([[0,1],[-2,3]])
#c = np.array([-3,1])
#b = np.array([4,6])

## Infeasible Check
#A = np.mat([[5,3],[3,5],[4,-3]])
#c = np.array([5,2])
#b = np.array([15,15,-12])

#print simplex(c, A, b)
