#Timothy Hills
#Lab: Linear Optimization
import scipy as sp
import numpy as np
import cvxopt
from cvxopt import matrix, solvers
#Problem one

def Prob1():
	'''
	Problem 1
	'''
	c = matrix([-1.0,-2.0,-3.0,-4.0,-5.0])
	G = matrix([[5.0,1.0,-1.0,-1.0,0.0,0.0,0.0,0.0],[4.0,-2.0,2.0,0.0,-1.0,0.0,0.0,0.0],[3.0,3.0,-3.0,0.0,0.0,-1.0,0.0,0.0],[2.0,-4.0,4.0,0.0,0.0,0.0,-1.0,0.0],[1.0,5.0,-5.0,0.0,0.0,0.0,0.0,-1.0]])
	h = matrix([30.0,20.0,10.0,0.0,0.0,0.0,0.0,0.0])
	sol = solvers.lp(c,G,h)
	print sol['x']
	print "c.T*sol['x']=", c.T*sol['x']
	
def Prob2():
	'''
	Problem 2
	'''
	c = matrix([-6.0,-8.0,-5.0,-9.0])
	G = matrix([[2.0,1.0,-1.0,0.0,0.0,0.0],[1.0,3.0,0.0,-1.0,0.0,0.0],[1.0,1.0,0.0,0.0,-1.0,0.0],[3.0,2.0,0.0,0.0,0.0,-1.0]])
	h = matrix([5.0,3.0,0.0,0.0,0.0,0.0])
	A = matrix([[1.0],[1.0],[1.0],[0.0]])
	b = matrix([1.0])
	sol = solvers.lp(c,G,h,A,b)
	print sol['x']
	print "c.T*sol['x']=", c.T*sol['x']
	
def Prob3():
	'''
	Problem 3
	'''
	c = matrix([-2.0,1.0,-9.0])
	G = matrix([[-1.0,2.0,-1.0,0.0,0.0],[-1.0,0.0,0.0,-1.0,0.0],[3.0,2.0,0.0,0.0,-1.0]])
	h = matrix([3.0,2.0,0.0,0.0,0.0])
	sol = solvers.lp(c,G,h)
	print sol['x']
	print "c.T*sol['x']=", c.T*sol['x']-6
	
def Prob4():
	'''
	Problem 4
	'''
	c = matrix([-5.0,-4.0])
	G = matrix([[1.0,-2.0,-1.0,0.0],[1.0,-2.0,0.0,-1.0]])
	h = matrix([2.0,-9.0,0.0,0.0])
	sol = solvers.lp(c,G,h)
	print sol['status']
	'''
	The status should be infeasible
	'''
def Prob5():
	'''
	Problem 5
	'''
	c = matrix([-1.0,4])
	G = matrix([[-2.0,-1,-1,0],[1,-2,0,-1]])
	h = matrix([-1.0,-2,0,0])
	sol = solvers.lp(c,G,h)
	print sol['status']
	'''
	Solution is infeasible, unbounded
	'''
	
def Prob6and7():
	'''
	Feel free to look at the code to determine if it is correct (problem 6)
	This is a problem that maximizes profit for a factory that produces
	steel at three different grades. In this approach, we will transpose G.
	'''
	c = matrix([-2.5*.8+1,-2.5*.05+.75,-2.5*.02+4.25,-2.5*.05+2.75, -2.5*.08+2.10, -1.8*.93+1, -1.8*.05+.75, 4.25, 2.75, -1.8*.02+2.10, -1.3*.99+1, -1.3*.01+.75, 4.25, 2.75, 2.10])
	G = matrix([[.80, .05, .02, .05, .08, 0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,.93,.05,0,0,.02, 0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,.99,.01, 0,0,0], [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0],[0,1,0,0,0,0,1,0,0,0,0,1,0,0,0],[0,0,1,0,0,0,0,1,0,0,0,0,1,0,0],[0,0,0,1,0,0,0,0,1,0,0,0,0,1,0],[0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],[-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1],[8*.8, 8*.05, 8*.02, 8*.05, 8*.08, 6*.93, 6*.05, 0, 0, 6*.02, 2*.99, 2*.01, 0, 0, 0]])
	h = matrix([5000.0, 15600, 8500, 15000, 10000, 2000, 3500, 10000,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,120000.0])
	sol = solvers.lp(c,G.T,h)
	x = sol['x']
	Af = x[0:5]
	Bf = x[5:10]
	Cf = x[10:]
	A = .8*Af[0]+.05*Af[1]+.02*Af[2]+.05*Af[3]+.08*Af[4]
	B = .93*Bf[0]+.05*Bf[1]+.02*Bf[4]
	C = .99*Cf[0]+.01*Cf[1]
	Fe = Af[0]+Bf[0]+Cf[0]
	Car = Af[1]+Bf[1]+Cf[1]
	Mn = Af[2]+Bf[2]+Cf[2]
	Ni = Af[3]+Bf[3]+Cf[3]
	Cr = Af[4]+Bf[4]+Cf[4]
	
	print "Prodution optimization=", sol['x']
	print "Where the metals are grouped together by grade (A,B,C).T,"
	print "and A,B,C have the metals ordered as Fe, C, Mn, Ni, Cr"
	print "The maximized profit given the constraints is ($) = ", -c.T*sol['x']
	print "The total amount of each grade produced (in lbs.)=" 
	print "A=",A
	print "B=", B
	print "C=", C
	print "The total amount of each metal needed for such production is (in lbs.):"
	print "Iron=", Fe
	print "Carbon=", Car
	print "Manganese=", Mn
	print "Nickel=", Ni
	print "Chromium=", Cr
	'''
	Profit = $11200
	A = 5000
	B = 8137.5
	C = .00003
	'''
	
def Prob8():
	'''
	From Prob6and7() we have changed the quantities of demand, suppliers and factory capacity
	separately and in combinations of such changes. The results are provided in the end
	'''
	c = matrix([-2.5*.8+.95,-2.5*.07+.75,-2.5*.02+4.25,-2.5*.03+2.75, -2.5*.08+2.10, -1.8*.93+.95, -1.8*.07+.75, 4.25, 2.75, -1.8*.02+2.10, -1.3*.99+.95, -1.3*.01+.75, 4.25, 2.75, 2.10])
	G = matrix([[.80, .05, .02, .05, .08, 0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,.93,.05,0,0,.02, 0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,.99,.01, 0,0,0], [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0],[0,1,0,0,0,0,1,0,0,0,0,1,0,0,0],[0,0,1,0,0,0,0,1,0,0,0,0,1,0,0],[0,0,0,1,0,0,0,0,1,0,0,0,0,1,0],[0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],[-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1],[8*.8, 8*.05, 8*.02, 8*.05, 8*.08, 6*.93, 6*.05, 0, 0, 6*.02, 2*.99, 2*.01, 0, 0, 0]])
	h = matrix([9000.0, 15600, 8500, 15000, 10000, 2000, 3500, 10000,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,126000.0])
	sol = solvers.lp(c,G.T,h)
	x = sol['x']
	Af = x[0:5]
	Bf = x[5:10]
	Cf = x[10:]
	A = .8*Af[0]+.05*Af[1]+.02*Af[2]+.05*Af[3]+.08*Af[4]
	B = .93*Bf[0]+.05*Bf[1]+.02*Bf[4]
	C = .99*Cf[0]+.01*Cf[1]
	Fe = Af[0]+Bf[0]+Cf[0]
	Car = Af[1]+Bf[1]+Cf[1]
	Mn = Af[2]+Bf[2]+Cf[2]
	Ni = Af[3]+Bf[3]+Cf[3]
	Cr = Af[4]+Bf[4]+Cf[4]
	
	print "Prodution optimization=", sol['x']
	print "Where the metals are grouped together by grade (A,B,C).T,"
	print "and A,B,C have the metals ordered as Fe, C, Mn, Ni, Cr"
	print "The maximized profit given the constraints is ($) = ", -c.T*sol['x']
	print "The total amount of each grade produced (in lbs.)=" 
	print "A=",A
	print "B=", B
	print "C=", C
	print "The total amount of each metal needed for such production is (in lbs.):"
	print "Iron=", Fe
	print "Carbon=", Car
	print "Manganese=", Mn
	print "Nickel=", Ni
	print "Chromium=", Cr
	'''
	As is noted, opening up the market to an increase demand in grade A, a 5 cent cheaper price
	in Iron and adding one hour to each work day will increase profits by about $3300.
	The other combinations did not affect the profit significantly. New profit = $14500
	'''
def Prob9(X,y):
	'''
	Solves the Chebyshev spproximation, that is, liinear regression using the infinity norm
	minimizing infnorm(XB-y), REMEMBER to enter y as a column vector and to cast X and y as floats
	'''	
	s = sp.shape(X)[1]
	t = sp.shape(X)[0]
	cn = sp.zeros(s+1)
	cn[0] = 1
	c = matrix(cn)
	print c
	en = -1.0*sp.ones(2*t)
	e = matrix(en)
	Xdn = sp.vstack([X,-X])
	Xd = matrix(Xdn)
	G = matrix([[e],[Xd]])
	print G
	y = matrix(y)
	hn = sp.vstack([y,-y])
	h = matrix(hn)
	print h
	return c,G,h
	sol = solvers.lp(c,G,h)
	print "t, Beta =", sol['x']	