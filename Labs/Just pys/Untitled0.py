# -*- coding: utf-8 -*-
# <nbformat>3</nbformat>

# <headingcell level=2>

# Genearal Descent (steepest)

# <codecell>

run GeneralDescent.py

# <codecell>

sp.set_printoptions(suppress=True, precision = 5)
A = sp.rand(8,5)
b = sp.rand(8)
x = sp.ones(5)
my_ans_steepest = steepest_descent(A, b, x)
my_ans_conj = conjugate_gradient(A, b, x)
least_squares_ans = sp.mat(la.lstsq(A,b)[0]).T
my_ans_steepest - least_squares_ans

# <codecell>

runProblem2(5,3)

# <headingcell level=2>

# Linear Programming

# <codecell>

run Linear_Programming.py

# <codecell>

problem_1()

# <codecell>

problem_2()

# <codecell>

problem_3()

# <codecell>

problem_4()

# <codecell>

problem_5()

# <codecell>

problem_6()

# <codecell>

X = sp.randn(8,8)
y = sp.randn(8)

# <codecell>

problem_9(X, y)

# <codecell>

problem_10(X, y)

# <codecell>

problem_11(X, y)

# <codecell>

vec1 = lineData(40)
problem_12(vec1[0], vec1[1])

# <codecell>

vec2 = ellipseData(40)
problem_13(vec2[0], vec2[1])

# <headingcell level=2>

# Simplex

# <codecell>

run Simplex.py

# <codecell>

# Easy check
A = np.mat([[-4,-2],[-2,3],[1.,0]])
c = np.array([1,2.])
b = np.array([-8,6,3.])
print simplex(c, A, b)

# <codecell>

# Unbounded Check
A = np.mat([[0,1],[-2,3]])
c = np.array([-3,1])
b = np.array([4,6])
print simplex(c, A, b)

# <codecell>

# Infeasible Check
A = np.mat([[5,3],[3,5],[4,-3]])
c = np.array([5,2])
b = np.array([15,15,-12])
print simplex(c, A, b)

# <headingcell level=2>

# Constrained Optimization

# <codecell>

run Constrained_Opt.py

# <codecell>

problem_1()

# <codecell>

problem_2_1()

# <codecell>

problem_2_2()

# <codecell>

problem_2_3()

# <codecell>

problem_4()

# <codecell>

problem_5()

# <codecell>


